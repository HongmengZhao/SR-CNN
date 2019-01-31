function center=HOG_kmeans(I_ref,I_index)
% Kmeans Clustering Using  HOG Features.
% I_ref:The Reference SCI
% I_index: The Corresponding Text Segmentation Index Map
% center:Clustering Center Matrix

%% HOG Feature Calculation
[m1,n1]=size(I_ref);
m2=floor(m1/8);n2=floor(n1/8);
[features, ~] = extractHOGFeatures(I_ref);
features=reshape(features,(m2-1),(n2-1),36);
%% K-Means
[x,y]=meshgrid(1:(n2-1),1:(m2-1));
im(:,:,1)=x;
im(:,:,2)=y;
im(:,:,3:38)=features;
im = reshape(im, (n2-1)*(m2-1), 38);
k = 10;
times = 5;
opts = statset('Display','final','MaxIter',1000);
[Idx,C,sumD,D]= kmeans(im, k, 'Options',opts);
for i = 2:times,
    [Idx_cur,C_cur,sumD_cur,D_cur] = kmeans(im, k, 'Options',opts);
    if (sum(sumD_cur) < sum(sumD)),
        D = D_cur;
        C = C_cur;
        sumD = sumD_cur;
        Idx=Idx_cur;
    end
end
c=zeros(k,4);
for C_i=1:k
    axis_x=round(((C(C_i,1)-1)*8+16+C(C_i,1)*8+16)/2);
    axis_y=round(((C(C_i,2)-1)*8+16+C(C_i,2)*8+16)/2);
    %% Adjust coordinates according to patches' size
    if axis_x+112>n1
        axis_x=n1-112;
    end
    if axis_x<=112
        axis_x=113;
    end
    if axis_y+112>m1
        axis_y=m1-112;
    end
    if axis_y<=112
        axis_y=113;
    end
    c(C_i,1)=axis_x-112;
    c(C_i,2)=axis_y-112;
    c(C_i,3)=numel(find(I_index(axis_y-112:axis_y+112,axis_x-112:axis_x+112)~=0))/(224*224);
    if numel(find(I_index(axis_y-112:axis_y+112,axis_x-112:axis_x+112)~=0))>=(224*224*0.45);
        c(C_i,4)=1;
    else
        c(C_i,4)=0;
    end
end
[~,rank]=sort(c(:,3),'descend');
c=c(rank,:);
center=zeros(6,2);

index1=find(c(:,4)==1);%textual center
index2=find(c(:,4)==0);%pictorial center
length=size(index1,1);
center1(1:length,1:2)=c(index1,1:2);
center2(1:k-length,1:2)=c(index2,1:2);

if length == 0
    center(1:3,:)=c(1:3,1:2);
    center(4:6,:)=center2(k-length-3:k-length-1,1:2);
end

if length == 1
    center(1:3,:)=repmat(center1(1,1:2),3,1);
    center(4:6,:)=center2(k-length-3:k-length-1,1:2);
end
if length == 2
    center(1:2,:)=center1(1:2,1:2);
    center(3,:)=center1(1,1:2);
    center(4:6,:)=center2(k-length-3:k-length-1,1:2);
end
if length >= 3 && length <= 7
    center(1:3,:)=center1(1:3,1:2);
    center(4:6,:)=center2(k-length-3:k-length-1,1:2);
end
if numel(index1) == 8
    center(1:3,:)=center1(1:3,1:2);
    center(4:5,:)=center2(1:2,1:2);
    center(6,:)=center2(1,1:2);
end
if numel(index1) == 9
    center(1:3,:)=center1(1:3,1:2);
    center(4:6,:)=repmat(center2(1,1:2),3,1);    
end
end