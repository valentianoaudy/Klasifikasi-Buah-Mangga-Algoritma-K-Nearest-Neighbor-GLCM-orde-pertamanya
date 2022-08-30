clear;clc;close all
%pelatihan
// folder data
cd('...');
datasetku ={'M1';'M2';'M3'};
jmlkls = length(datasetku);
for n=1:jmlkls
    cd(char(datasetku(n)));
    datacitra = dir('*.jpg');
    jmldata = length(datacitra);
    for i=1:jmldata
        namafile = datacitra(i).name;
        citrai = rgb2gray(imread(namafile));
        H = imhist(citrai)';
        H = H/sum(H);
        I = (0:255);
        CiriMEAN = I*H';
        CiriVAR = (I-CiriMEAN).^2*H';
        CiriSKEW = (I-CiriMEAN).^3*H'/CiriVAR^1.5;
        CiriKURT = (I-CiriMEAN).^4*H'/CiriVAR^2-3;
        %GLCM
        data_training(i+jmldata*(n-1),1) = CiriMEAN;
%         data_training(i+jmldata*(n-1),2) = CiriVAR;
%         data_training(i+jmldata*(n-1),3) = CiriSKEW;
%         data_training(i+jmldata*(n-1),4) = CiriKURT;
    
        kelas(i+jmldata*(n-1))=n;
    end
    cd('..');
end
%pengujian
model = fitcknn(data_training,kelas','NumNeighbors',1,'Standardize',1); %Model KNN
cd('Tes');
for j=1:9
    nama=sprintf('M%d.jpg',j)
    a = rgb2gray(imread(nama));
    H = imhist(a)';
    H = H/sum(H);
    I = [0:255];
    CiriMEAN = I*H';
    CiriVAR = (I-CiriMEAN).^2*H';
    CiriSKEW = (I-CiriMEAN).^3*H'/CiriVAR^1.5;
    CiriKURT = (I-CiriMEAN).^4*H'/CiriVAR^2-3;
    data_testing(j,1)= CiriMEAN;
%     data_testing(j,2)= CiriVAR;
%     data_testing(j,3)= CiriSKEW;
%     data_testing(j,4)= CiriKURT;
  
    target(1)=1;
    target(2)=1;
    target(3)=1;
    target(4)=2;
    target(5)=2;
    target(6)=2;
    target(7)=3;
    target(8)=3;
    target(9)=3;
    
    klasifikasi(j) = model.predict(data_testing(j,:)); %Melakukan prediksi Model
    if klasifikasi(j)==target(j)
        hasil(j) = {'Benar'};
    else
        hasil(j) = {'Salah'};
    end
end
[{'Mean','Variance','Skewness','Kurtosis','Target','Kelas','Hasil'};
num2cell([data_testing target' klasifikasi']) hasil']
    
cm = confusionmat(target',klasifikasi')
akurasi = sum(diag(cm))/sum(sum(cm))