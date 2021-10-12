% Matlab code for generating satellite image mask.
% Satellie imagery 'IM.jpg'.  Mask written to 'mask_IM.jpg'.

close all
clear all

format long

file='CHN0058_17_3000.jpg'; % Generate mask for this file.
im=imread(file);
mask=zeros(size(im));

figure(1)
clf
imshow(file);
hold on;

n=input('Number of regions: '); % Number of separate non-masked out blobs in image.

for i=1:n,
    h{i}=drawpolygon;
    fill(h{i}.Position(:,1),h{i}.Position(:,2),'w');
end;

figure(2)
clf

for i=1:n,
    new_mask=poly2mask(h{i}.Position(:,1),h{i}.Position(:,2),size(im,1),size(im, ...
                                                      2));
    mask=max(mask,new_mask);
end;
imshow(mask);

imwrite(squeeze(mask(:,:,1)),['mask_' file],'jpg');