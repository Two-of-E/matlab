img = imread('F:\MATLAB\touxiang.jpg');
img1=img(:,:,1);%��Ӧrgb
img2=img(:,:,2);
img3=img(:,:,3);

%̨ͬ�˲�����
            % figure,imshow(img1);
            % title('Original Image1');
            % figure,imshow(img2);
            % title('Original Image2');
            % figure,imshow(img3);
            % title('Original Image3');
            % figure,imhist(img1);
            % title('hist Image1');
            % figure,imhist(img2);
            % title('hist Image2');
            % figure,imhist(img3);
            % title('hist Image3');
% F = fftshift(fft2(img3));  %��ͼ�������Ƶ��м�
% imF = log10(abs(F)+1);
% figure,imshow(imF);
% F1 = fftshift(fft2(img2));  %��ͼ�������Ƶ��м�
% imF1 = log10(abs(F1)+1);
% figure,imshow(imF1);

% ����һ����˹�˲���
f_high = 1.0;
f_low = 0.8;
% �õ�һ����˹��ͨ�˲�����fspecial����һ���˲����������������Ǳ�׼��
gauss_low_filter = fspecial('gaussian', [7 7],0.8);% 0.4
%size����˲���������к���
matsize = size(gauss_low_filter);
% ����̬ͬ�˲���Ҫ�˳���Ƶ����,
% �������ǵð������ͨ�˲���ת����һ����ͨ�˲���.
% f_high �� f_low �ǿ��������ͨ�˲�����̬�Ĳ���.
% zeros����һ���յľ���
gauss_high_filter = zeros(matsize);
gauss_high_filter(ceil(matsize(1,1)/2) , ceil(matsize(1,2)/2)) = 1.0;
% figure,freqz2(gauss_high_filter);
% figure,freqz2(gauss_low_filter);
gauss_high_filter = f_high*gauss_high_filter - (f_high-f_low)*gauss_low_filter;

% ��ʾ��ͨ�˲��ڵ�Ƶ����Ӧ
figure,freqz2(gauss_high_filter);
colormap(jet(64));%ת��Ϊ64�Ҷ�3ɫ
% ���ö����任�������ͷ���ⲿ�ַֿ�
log_img1 = log(double(img1));
log_img2 = log(double(img2));
log_img3 = log(double(img3));
% ����˹��ͨ�˲��������ת�����ͼ����
high_log_part1 = imfilter(log_img1, gauss_high_filter, 'symmetric', 'conv');
high_log_part2 = imfilter(log_img2, gauss_high_filter, 'symmetric', 'conv');
high_log_part3 = imfilter(log_img3, gauss_high_filter, 'symmetric', 'conv');
% ��ʾ������ͼ��
% figure,imshow(high_log_part1);
% title('High Frequency Part1');
% figure,imshow(high_log_part2);
% title('High Frequency Part2');
% figure,imshow(high_log_part3);
% title('High Frequency Part3');
% ���ڱ������ͼ���Ǿ��������任��,���������ݱ任��ͼ��ָ�����
high_part1 = exp(high_log_part1);
high_part2 = exp(high_log_part2);
high_part3 = exp(high_log_part3);

high_part3(isnan(high_part3)==1) = 0;
[inf_r1, inf_c1] = find(high_part3==0);
[inf_r2, inf_c2] = find(high_part3==inf);

inf_r=[inf_r1;inf_r2];
inf_c=[inf_c1;inf_c2];

[m,n]=size(img1);
[a,b]=size(inf_r);

high_part3_ave=high_part3;
high_part3_ave(inf_r,inf_c)= 0;
ave_test3=mean(high_part3_ave(:));
%ave_test3 = mean(mean(high_part3_ave));
ave=ave_test3*(m*m)/(m*m-a);


high_part3(inf_r,inf_c)= ave;

minv1 = min(min(high_part1));
maxv1 = max(max(high_part1));
minv2 = min(min(high_part2));
maxv2 = max(max(high_part2));
minv3 = min(min(high_part3));
maxv3 = max(max(high_part3));
% �õ��Ľ��ͼ��
imm1=(high_part1-minv1)/(maxv1-minv1);
imm2=(high_part2-minv2)/(maxv2-minv2);
imm3=(high_part3-minv3)/(maxv3-minv3);
% figure,imshow(imm1);
% title('Result Image1');
            % figure,imshow(imm2);
            % title('Result Image2');
            % figure,imshow(imm3);
            % title('Result Image3');
img_hazy=cat(3,imm1,imm2,imm3);
% img_hazy=cat(3,imm1,imm2,double(img3)/340);



figure,imshow(img_hazy);
title('imghazy Image');


