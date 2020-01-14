clear all;
close all;
clc;

image=imread('./data/JPEGImages/generate0_61.jpg');
imgray=rgb2gray(image);



% for i=1:size(imgray,1)
%     for j=1:size(imgray,2)
%         
%         if imgray(i,j)==0
%             imgray(i,j)=255;
%             
%         end
%         
%     end
% end

imshow(imgray);