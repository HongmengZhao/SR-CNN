function score = test( test_data )

%% Load Net
caffe.set_mode_cpu();
caffe.reset_all();
deploy = '.\test\test.prototxt';
caffe.reset_all();
caffe_model = ['.\test\test.caffemodel'];
net = caffe.Net(deploy, caffe_model, 'test');
batches = cell(1, 1);
patch_score=[];
for i=1:size(test_data,4)
    aaa(:,:,:,1) = single(test_data(:,:,:,i));
    batches{i} = aaa;
    res{i} = net.forward(batches(i)); %Forward propagation results
    patch_score(i,1)=cell2mat(res{1,i}) ;
end
score=mean(patch_score);
end