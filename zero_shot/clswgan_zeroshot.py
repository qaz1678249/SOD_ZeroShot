import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split


file_name = ''

noise_dim = 32
cnnfeature_dim = 2048
attribute_dim = 66

learning_rate = 0.0001
epochs = 2000

# train using gpu
cuda = True

# pretrain classifier parameters
pre_cls_epochs = 20
batch_size = 64

# parameters for generator
g_fc1 = 4096
g_fc2 = 2048

# parameters for discriminator
d_fc1 = 4096

pen_weight = 10
cls_weight = 1

gzsl = True
gen_samples = 50

class Classifier(nn.Module):
    def __init__(self, in_dim, num_cls):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_cls)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.g_fc1 = nn.Linear(noise_dim + attribute_dim, g_fc1)
        self.g_fc2 = nn.Linear(g_fc1, g_fc2)
        self.g_fc3 = nn.Linear(g_fc2, cnnfeature_dim)

    def forward(self, noise, attributes):
        x = torch.cat((noise, attributes), 1)
        x = F.leaky_relu(self.g_fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.g_fc2(x), negative_slope=0.2)
        x = F.relu(self.g_fc3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.d_fc1 = nn.Linear(cnnfeature_dim + attribute_dim, d_fc1)
        self.d_fc2 = nn.Linear(d_fc1, 1)

    def forward(self, cnn_features, attributes):
        x = torch.cat((cnn_features, attributes), 1)
        x = F.leaky_relu(self.d_fc1(x), negative_slope=0.2)
        x = self.d_fc2(x)
        return x


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * pen_weight
    return gradient_penalty


def train_classifier(cls, dataset, split):
    """
    train the softmax classifier using given dataset
    :param cls: softmax classifier model
    :param data: torch.utils.data.Dataset the dataset to be trained on
    :param split: list to split dataset into training set and valid set
    :return: a trained classifier model.
    """
    if cuda:
        cls.cuda()

    [trainset, validset] = random_split(dataset, split)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(validset, batch_size=split[1], shuffle=False)
    optimizer = optim.SGD(cls.parameters(), lr=0.0001, momentum=0.9)

    cls.train()
    train_loss, train_accu, valid_loss, valid_accu = [], [], [], []
    i = 0
    for epoch in range(pre_cls_epochs):
        for features, labels in trainloader:
            # send to gpu
            if cuda:
                features, labels = features.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = cls(features)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            predictions = outputs.data.max(1)[1]
            accuracy = np.sum(predictions.cpu().numpy() == labels.cpu().numpy()) / batch_size * 100
            train_accu.append(accuracy)
            if i % 100 == 0:
                v_correct = 0
                for v_images, v_labels in valloader:
                    if cuda:
                        v_images, v_labels = v_images.cuda(), v_labels.cuda()
                    v_outputs = cls(v_images)
                    val_loss = F.cross_entropy(v_outputs, v_labels)
                    valid_loss.append(val_loss.item())
                    v_predictions = v_outputs.data.max(1)[1]
                    v_correct += v_predictions.eq(v_labels.data).sum()

                    v_accuracy = v_correct.cpu().numpy() / split[1] * 100
                valid_accu.append(v_accuracy)
                # print('Train step: {}\tTrain loss:{:.3f}\tValid loss:{:.3f}\tTrain accuracy: {:.3f}\tValid accuracy: {:.3f}'.format(i,
                #                                                                                                  loss.item(),
                #                                                                                                  val_loss.item(),
                #                                                                                                  accuracy,
                #                                                                                                  v_accuracy))
                if len(valid_loss)>1:
                    if valid_loss[-1]>valid_loss[-2]:
                        break
            i += 1
    return cls


def sample():
    batch_feature, batch_att, batch_label = next(iter(trainloader))
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(batch_label)


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.empty(nclass * num, cnnfeature_dim).type(torch.FloatTensor)
    syn_label = torch.empty(nclass * num).type(torch.LongTensor)
    syn_att = torch.empty(num, attribute_dim).type(torch.FloatTensor)
    syn_noise = torch.empty(num, noise_dim).type(torch.FloatTensor)
    if cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label


if __name__ =="__main__":
    # load data
    data = np.load('features_attributes_labels.npz')

    cnn_features = data['features']
    attributes = data['attributes']
    labels = data['labels']
    # labels = np.argmax(labels, axis=1)
    unseen_features = data['unseen_features']
    unseen_attributes = data['unseen_attributes']
    unseen_labels = data['unseen_labels']

    num_samples = labels.shape[0]
    split = [int(np.ceil(num_samples * 0.9)), num_samples - int(np.ceil(num_samples * 0.9))]
    num_seen_cls = np.max(labels)
    num_unseen_cls = 1
    # num_unseen_cls = unseen_cls.shape[0]

    pre_cls_dataset = TensorDataset(torch.Tensor(cnn_features), torch.Tensor(labels).type(torch.LongTensor))
    dataset = TensorDataset(torch.Tensor(cnn_features), torch.Tensor(attributes), torch.Tensor(labels).type(torch.LongTensor))
    unseen_dataset = TensorDataset(torch.Tensor(unseen_features), torch.Tensor(unseen_attributes), torch.Tensor(unseen_labels).type(torch.LongTensor))
    [trainset, validset] = random_split(dataset, split)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(validset, batch_size=split[1], shuffle=False)
    unseenloader = DataLoader(unseen_dataset, unseen_labels.shape[0], shuffle=False)

    pre_cls = Classifier(cnnfeature_dim, num_seen_cls)
    print(pre_cls)
    train_classifier(pre_cls, pre_cls_dataset, split)
    torch.save(pre_cls, 'pre_trained_cls')
    # OR LOAD PRETRAINED CLASSIFIER
    # pre_cls = torch.load('pre_trained_cls')
    pre_cls.eval()



    # train GAN
    # dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    input_res = torch.empty(batch_size, cnnfeature_dim).type(torch.FloatTensor)
    input_att = torch.empty(batch_size, attribute_dim).type(torch.FloatTensor)
    noise = torch.empty(batch_size, noise_dim).type(torch.FloatTensor)
    one = torch.Tensor([1]).type(torch.FloatTensor)
    mone = one * -1
    input_label = torch.empty(batch_size).type(torch.LongTensor)
    # final_label = torch.FloatTensor(batch_size, num_all_cls)

    net_G = Generator()
    print(net_G)
    net_D = Discriminator()
    print(net_D)

    cls_criterion = nn.CrossEntropyLoss()

    if cuda:
        net_D.cuda()
        net_G.cuda()
        input_res = input_res.cuda()
        noise, input_att = noise.cuda(), input_att.cuda()
        one = one.cuda()
        mone = mone.cuda()
        cls_criterion.cuda()
        input_label = input_label.cuda()

    for p in pre_cls.parameters():  # set requires_grad to False
        p.requires_grad = False

    optimizerD = optim.Adam(net_D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(net_G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    vs_acc, vu_acc, H = [], [], []

    for epoch in range(epochs):
        FP = 0
        mean_lossD = 0
        mean_lossG = 0
        for i in range(0, num_samples, batch_size):

            for p in net_D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for iter_d in range(5):
                sample()
                net_D.zero_grad()
                # train with realG
                # sample a mini-batch
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                criticD_real = net_D(input_resv, input_attv)
                criticD_real = criticD_real.mean()
                criticD_real.backward(mone)

                # train with fakeG
                noise.normal_(0, 1)
                noisev = Variable(noise)
                fake = net_G(noisev, input_attv)
                fake_norm = fake.data[0].norm()
                sparse_fake = fake.data[0].eq(0).sum()
                criticD_fake = net_D(fake.detach(), input_attv)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(one)

                # gradient penalty
                gradient_penalty = calc_gradient_penalty(net_D, input_res, fake.data, input_att)
                gradient_penalty.backward()

                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()

            ############################
            # (2) Update G network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in net_D.parameters():  # reset requires_grad
                p.requires_grad = False  # avoid computation

            net_G.zero_grad()
            input_attv = Variable(input_att)
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = net_G(noisev, input_attv)
            criticG_fake = net_D(fake, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake
            # classification loss
            c_errG = cls_criterion(pre_cls(fake), Variable(input_label))
            errG = G_cost + cls_weight * c_errG
            errG.backward()
            optimizerG.step()

        mean_lossG /= num_samples / batch_size
        mean_lossD /= num_samples / batch_size
        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
              % (epoch, epochs, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG.item()))

        # evaluate the model, set G to evaluation mode
        net_G.eval()
        # Generalized zero-shot learning
        if gzsl:
            syn_feature, syn_label = generate_syn_feature(net_G, torch.Tensor(unseen_labels).type(torch.LongTensor), torch.Tensor(unseen_attributes).type(torch.FloatTensor), gen_samples)
            train_X = torch.cat((torch.Tensor(cnn_features), syn_feature), 0)
            train_Y = torch.cat((torch.Tensor(labels).type(torch.LongTensor), syn_label), 0)
            clsdataset = TensorDataset(train_X, train_Y)
            nclass = num_seen_cls + num_unseen_cls
            cls = Classifier(cnnfeature_dim, nclass)
            n_tr = int(np.ceil(0.9 * (num_samples + gen_samples * unseen_labels.shape[0])))
            n_v = num_samples + gen_samples * unseen_labels.shape[0] - n_tr
            split_gzsl = [n_tr, n_v]
            train_classifier(cls, clsdataset, split_gzsl)
            cls.eval()
            s_correct = 0
            for s_images, s_attributes, s_labels in valloader:
                if cuda:
                    s_images, s_labels = s_images.cuda(), s_labels.cuda()
                s_outputs = cls(s_images)
                s_predictions = s_outputs.data.max(1)[1]
                s_correct += s_predictions.eq(s_labels.data).sum()

                s_accuracy = s_correct.cpu().numpy() / split[1] * 100
            vs_acc.append(s_accuracy)
            u_correct = 0
            for u_images, u_attributes, u_labels in unseenloader:
                if cuda:
                    u_images, u_labels = u_images.cuda(), u_labels.cuda()
                u_outputs = cls(u_images)
                u_predictions = u_outputs.data.max(1)[1]
                u_correct += u_predictions.eq(u_labels.data).sum()

                u_accuracy = u_correct.cpu().numpy() / unseen_labels.shape[0] * 100
            vu_acc.append(u_accuracy)

            h = 2*s_accuracy*u_accuracy/(s_accuracy+u_accuracy)
            H.append(h)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (u_accuracy, s_accuracy, h))
            torch.save(cls, 'pre_trained_cls')
        # # Zero-shot learning
        # else:
        #     syn_feature, syn_label = generate_syn_feature(net_G, unseen_labels, unseen_attributes, gen_samples)
        #     clsdataset = TensorDataset((syn_feature, syn_label))
        #     cls = Classifier(cnnfeature_dim, num_unseen_cls)
        #     split_zsl = [0.9 * gen_samples * num_unseen_cls, 0.1 * gen_samples * num_unseen_cls]
        #     train_classifier(cls, clsdataset, split_zsl)
        #     acc = cls.acc
        #     print('unseen class accuracy= ', acc)

        # reset G to training mode
        net_G.train()

        torch.save(net_D, 'net_D')
        torch.save(net_G, 'net_G')
