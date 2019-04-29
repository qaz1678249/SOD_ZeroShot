%% Set the parameters
param.eta = 1e-1;               %learning rate, should be tuned on the validation set
param.nepoch = 10;             %number of epochs, fixed
param.K = 10;                   %number of latent embeddings, should be tuned on the validation set
param.cls_emb = 'word2vec';     %name of class embedding to evaluate, you can also use 'glove', 'wordnet' and 'cont'

%% loading data
%load('data_CUB');
%[train_X, xval_mean, xval_variance, xval_max] = normalization(train_X);
[trainval_X, xtest_mean, xtest_variance, xtest_max] = normalization(trainval_X);
%val_X = normalization(val_X, xval_mean, xval_variance, xval_max);
test_X = normalization(test_X, xtest_mean, xtest_variance, xtest_max);

%% Train the model using train+val data with the eta and K selected on the validation set
disp('Train the model...');
disp(['K=' num2str(param.K) ', eta=' num2str(param.eta) ', nepoch=' num2str(param.nepoch)]);
%W = latEm_train(trainval_X, trainval_labels, trainval_Y(param.cls_emb), param.eta, param.nepoch, param.K);
W = latEm_train(trainval_X, trainval_labels, trainval_att, param.eta, param.nepoch, param.K);
%acc_test = latEm_test(W, test_X, test_Y(param.cls_emb), test_labels);
acc_test = latEm_test(W, test_X, test_att, test_labels);
disp(['Mean class accuracy=' num2str(acc_test)]);
