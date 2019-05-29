function center=HOG_kmeans(I_ref,I_index,k)
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
opts = statset('Display','final','MaxIter',1000);
[~,C]= kmeans(im, k, 'Options',opts);
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
    center(1:k/2,:)=c(1:k/2,1:2);
    center(k/2+1:k,:)=center2(k-length-k/2:k-length-1,1:2);
end
if length <k/2 && length>0
    center(1:length,:)=center1(1:length,1:2);
    center(length+1:k/2,:)=repmat(center1(1,1:2),k/2-length,1);
    center(k/2+1:k,:)=center2(k-length-k/2:k-length-1,1:2);
end
if length >=k/2 && length<k
    center(1:k/2,:)=center1(1:k/2,1:2);
    center(k/2+1:k/2+length,:)=center2(1:length,1:2);
    center(length+k/2+1:k,:)=repmat(center2(length,1:2),k/2-length,1);
end
if length ==k
    center(1:k/2,:)=center1(1:k/2,1:2);
    center(k/2+1:k,:)=c(k/2+1:k,1:2);
end
end