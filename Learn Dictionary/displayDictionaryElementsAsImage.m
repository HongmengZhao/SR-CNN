function I = displayDictionaryElementsAsImage(D, numRows, numCols,X,Y,sortVarFlag)
% function I = displayDictionaryElementsAsImage(D, numRows, numCols, X,Y)
% displays the dictionary atoms as blocks. For activation, the dictionary D
% should be given, as also the number of rows (numRows) and columns
% (numCols) for the atoms to be displayed. X and Y are the dimensions of
% each atom.


borderSize = 1;%边界宽度
columnScanFlag = 1;%列搜索标志
strechEachVecFlag = 1;%拉长每个向量的标志
showImFlag = 1;%显示图像的标志
%检查一下系数是否缺省
if (length(who('X'))==0)
    X = 8;
    Y = 8;
end
if (length(who('sortVarFlag'))==0)
    sortVarFlag = 1;
end

numElems = size(D,2);
if (length(who('numRows'))==0)
    numRows = floor(sqrt(numElems));
    numCols = numRows;
end
if (length(who('strechEachVecFlag'))==0) 
    strechEachVecFlag = 0;
end
if (length(who('showImFlag'))==0) 
    showImFlag = 1;
end

%%% sort the elements, if necessary.
%%% construct the image to display (I)
sizeForEachImage = sqrt(size(D,1))+borderSize;
I = zeros(sizeForEachImage*numRows+borderSize,sizeForEachImage*numCols+borderSize,3);
%%% fill all this image in blue
I(:,:,1) = 0;%min(min(D));
I(:,:,2) = 0; %min(min(D));
I(:,:,3) = 1; %max(max(D));

%%% now fill the image squares with the elements (in row scan or column
%%% scan).
if (strechEachVecFlag)
    for counter = 1:size(D,2)
        D(:,counter) = D(:,counter)-min(D(:,counter));%令字典每一列减去这一列最小的元素
        if (max(D(:,counter)))%检查这一列元素是否全部是0
            D(:,counter) = D(:,counter)./max(D(:,counter));%令字典这一列的每个元素除以最大的元素
        end
    end
end


if (sortVarFlag)
    vars = var(D);%返回每一列方差的无偏估计值的行向量（即分母为N-1）
    [V,indices] = sort(vars');%对列向量进行升序排列
    indices = fliplr(indices);%对矩阵进行左右翻转，对列向量不做操作
    D = [D(:,1:sortVarFlag-1),D(:,indices+sortVarFlag-1)];
    signs = sign(D(1,:));%当x<0时，sign(x)=-1；当x=0时，sign(x)=0; 当x>0时，sign(x)=1。
    signs(find(signs==0)) = 1;
    D = D.*repmat(signs,size(D,1),1);%如果A是一个3x4x5的矩阵，有B = repmat(A,2,3)则最后的矩阵是6x12x5
    D = D(:,1:numRows*numCols);
end

counter=1;
for j = 1:numRows
    for i = 1:numCols
%         if (strechEachVecFlag)
%             D(:,counter) = D(:,counter)-min(D(:,counter));
%             D(:,counter) = D(:,counter)./max(D(:,counter));
%         end
%         if (columnScanFlag==1)
%             I(borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,1)=reshape(D(:,counter),8,8);
%             I(borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,2)=reshape(D(:,counter),8,8);
%             I(borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,3)=reshape(D(:,counter),8,8);
%         else
            % Go in Column Scan:
            I(borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,1)=reshape(D(:,counter),X,Y);
            I(borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,2)=reshape(D(:,counter),X,Y);
            I(borderSize+(j-1)*sizeForEachImage+1:j*sizeForEachImage,borderSize+(i-1)*sizeForEachImage+1:i*sizeForEachImage,3)=reshape(D(:,counter),X,Y);
%         end
        counter = counter+1;
    end
end

if (showImFlag) 
    I = I-min(min(min(I)));
    I = I./max(max(max(I)));
    imshow(I,[]);
end
