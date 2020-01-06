img = imread('F:\MATLAB\touxiang.jpg');
img1=img(:,:,1);%对应rgb
img2=img(:,:,2);
img3=img(:,:,3);

%同台滤波程序
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
% F = fftshift(fft2(img3));  %将图像中心移到中间
% imF = log10(abs(F)+1);
% figure,imshow(imF);
% F1 = fftshift(fft2(img2));  %将图像中心移到中间
% imF1 = log10(abs(F1)+1);
% figure,imshow(imF1);

% 构造一个高斯滤波器
f_high = 1.0;
f_low = 0.8;
% 得到一个高斯低通滤波器，fspecial构造一个滤波器，第三个参数是标准差
gauss_low_filter = fspecial('gaussian', [7 7],0.8);% 0.4
%size输出滤波器矩阵的行和列
matsize = size(gauss_low_filter);
% 由于同态滤波是要滤出高频部分,
% 所以我们得把这个低通滤波器转换成一个高通滤波器.
% f_high 和 f_low 是控制这个高通滤波器形态的参数.
% zeros创建一个空的矩阵
gauss_high_filter = zeros(matsize);
gauss_high_filter(ceil(matsize(1,1)/2) , ceil(matsize(1,2)/2)) = 1.0;
% figure,freqz2(gauss_high_filter);
% figure,freqz2(gauss_low_filter);
gauss_high_filter = f_high*gauss_high_filter - (f_high-f_low)*gauss_low_filter;

% 显示搞通滤波期的频率响应
figure,freqz2(gauss_high_filter);
colormap(jet(64));%转化为64灰度3色
% 利用对数变换将入射光和反射光部分分开
log_img1 = log(double(img1));
log_img2 = log(double(img2));
log_img3 = log(double(img3));
% 将高斯高通滤波器与对数转换后的图象卷积
high_log_part1 = imfilter(log_img1, gauss_high_filter, 'symmetric', 'conv');
high_log_part2 = imfilter(log_img2, gauss_high_filter, 'symmetric', 'conv');
high_log_part3 = imfilter(log_img3, gauss_high_filter, 'symmetric', 'conv');
% 显示卷积后的图象
% figure,imshow(high_log_part1);
% title('High Frequency Part1');
% figure,imshow(high_log_part2);
% title('High Frequency Part2');
% figure,imshow(high_log_part3);
% title('High Frequency Part3');
% 由于被处理的图象是经过对数变换的,我们再用幂变换将图象恢复过来
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
% 得到的结果图象
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


