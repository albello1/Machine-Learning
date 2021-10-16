load('Data_train.mat');
load('Data_test.mat');
% CORPUS_TRAIN=CORPUS_T.CORPUS_TRAIN;
% CORPUS_TEST=CORPUS_TT.CORPUS_TEST;
MV = ismissing(CORPUS_TRAIN_table);
[M, N] = size(MV);
for i=1:M
    for j=1:(N-1)
        if(MV(i, j) == 1)
            CORPUS_TRAIN(i, j) = nanmean(CORPUS_TRAIN(:, j));
        end
    end
end

MV = ismissing(CORPUS_TRAIN);
etiquetas = CORPUS_TRAIN_table.Properties.VariableNames;


[m,n]=size(CORPUS_TRAIN);

if n==22
    CORPUS_TRAIN(:,13)=[];
CORPUS_TRAIN(:,18-1)=[];
CORPUS_TRAIN(:,19-2)=[];
CORPUS_TRAIN(:,6)=[];
CORPUS_TRAIN(:,6)=[];
end
[m,n]=size(CORPUS_TRAIN);
ac1=0;
ac2=0;


for i=1:1:m
    if CORPUS_TRAIN(i,n)==1
        mat1(i,:)=CORPUS_TRAIN(i,:);
        ac1=ac1+1;
    elseif CORPUS_TRAIN(i,n)==2
        mat2(i-ac1,:)=CORPUS_TRAIN(i,:);
        ac2=ac2+1;
    else
        mat3(i-(ac2+ac1),:)=CORPUS_TRAIN(i,:);
        
    end
end

[m1,n1]=size(mat1);
[m2,n2]=size(mat2);
[m3,n3]=size(mat3);

MV = ismissing(CORPUS_TEST);
[M, N] = size(MV);
for i=1:M
    for j=1:(N-1)
        if(MV(i, j) == 1)
            CORPUS_TEST(i, j) = nanmean(CORPUS_TEST(:, j));
        end
    end
end

[m11,n11]=size(CORPUS_TEST);
if n11==22
    CORPUS_TEST(:,13)=[];
    CORPUS_TEST(:,18-1)=[];
    CORPUS_TEST(:,19-2)=[];
    CORPUS_TEST(:,6)=[];
    CORPUS_TEST(:,6)=[];
end
ac1=0;
ac2=0;
[m11,n11]=size(CORPUS_TEST);

for i=1:1:m11
    if CORPUS_TEST(i,n11)==1
        mat1t(i,:)=CORPUS_TEST(i,:);
        ac1=ac1+1;
    elseif CORPUS_TEST(i,n11)==2
        mat2t(i-ac1,:)=CORPUS_TEST(i,:);
        ac2=ac2+1;
    else
        mat3t(i-(ac2+ac1),:)=CORPUS_TEST(i,:);
        
    end
end

[m1t,n1t]=size(mat1t);
[m2t,n2t]=size(mat2t);
[m3t,n3t]=size(mat3t);


[m11,n11]=size(CORPUS_TEST);


 
outliers1 = isoutlier(mat1);
outliers2 = isoutlier(mat2);
outliers3 = isoutlier(mat3);
 

 
outliers = {outliers1,outliers2,outliers3}; %Celda para guardar las matrices
 

%Para clase 1:
cont=0; %Inicializo un contador para evitar que al perder filas de la matriz
       %se me generen errores
    
for i=1:length(outliers1)    
    if sum(int8(outliers1(i,:)))>4%Si el Nº de outliers es mayor que la    
                                  %condición...
        mat1(i-cont,:) = []; %... eliminamos la muestra
        cont=cont+1;%Incremento el contador 
    end
end   
%Para clase 2:
cont=0;
for i=1:length(outliers2)    
    if sum(int8(outliers2(i,:)))>7
 
        mat2(i-cont,:) = [];
        cont=cont+1;
    end
end   
%Para clase 3:
cont=0;
for i=1:length(outliers3)    
    if sum(int8(outliers3(i,:)))>8
        mat3(i-cont,:) = [];
        cont=cont+1;
    end
end   
 
%Reconstruyo la matriz data con los nuevos datos (Una vez tratados los
%outlaiers: 
data = [mat1;mat2;mat3]; 
CORPUS_TRAIN=data;
[m,n]=size(CORPUS_TRAIN);

matrizCorr= corr(CORPUS_TRAIN(:,1:n-1), CORPUS_TRAIN(:,1:n-1));


outliers1 = isoutlier(mat1t);
outliers2 = isoutlier(mat2t);
outliers3 = isoutlier(mat3t);

outliers = {outliers1,outliers2,outliers3}; %Celda para guardar las matrices



%Para clase 1:
cont=0; %Inicializo un contador para evitar que al perder filas de la matriz
       %se me generen errores
    
for i=1:length(outliers1)    
    if sum(int8(outliers1(i,:)))>4%Si el Nº de outliers es mayor que la    
                                  %condición...
        mat1t(i-cont,:) = []; %... eliminamos la muestra
        cont=cont+1;%Incremento el contador 
    end
end   
%Para clase 2:
cont=0;
for i=1:length(outliers2)    
    if sum(int8(outliers2(i,:)))>7
 
        mat2t(i-cont,:) = [];
        cont=cont+1;
    end
end   
%Para clase 3:
cont=0;
for i=1:length(outliers3)    
    if sum(int8(outliers3(i,:)))>8
        mat3t(i-cont,:) = [];
        cont=cont+1;
    end
end   
 
%Reconstruyo la matriz data con los nuevos datos (Una vez tratados los
%outlaiers: 
data = [mat1t;mat2t;mat3t]; 
CORPUS_TEST=data;
[m1,n2]=size(CORPUS_TEST);


[ranks,weights] = relieff(CORPUS_TRAIN(:,1:n-1),CORPUS_TRAIN(:,n),m); 



seleccion2=7;
seleccion22=7;
for j=1:seleccion2
    matBienTrainr(:,j)=CORPUS_TRAIN(:,ranks(1,j));
end

for h=1:seleccion22
    matBienTestr(:,h)=CORPUS_TEST(:,ranks(1,h));
end
TRdata=matBienTrainr;
TEdata=matBienTestr;
TEtag=CORPUS_TEST(:,end);
TRtag=CORPUS_TRAIN(:,end);

[me,ne] = size(TRdata);
[mt,nt] = size(TEdata);



 
%K-VECINOS 1
 mdvec1 = fitcknn(TRdata,TRtag, 'NumNeighbors',1);
 TEtag_pred = predict(mdvec1, TEdata);
 Error_K1=(sum(TEtag~=TEtag_pred)/length(TEtag));


 [ms,ns]=size(TEtag);
 
 
 %Sensibilidad
 aciertosSens=0;
 fallosSens=0;
 for i=1:ms
     if TEtag(i,1)==3
         if TEtag_pred(i,1)==3
             aciertosSens=aciertosSens+1;
         elseif TEtag_pred(i,1)==1
             fallosSens=fallosSens+1;
         else
         end
     end
 end
 
 sensibilidad=(aciertosSens/(aciertosSens+fallosSens))*100;
 
 
 %Especificidad
 aciertosEsp=0;
 fallosEsp=0;
 for i=1:ms
     if TEtag(i,1)==1
         if TEtag_pred(i,1)==1
             aciertosEsp=aciertosEsp+1;
         elseif TEtag_pred(i,1)==3
             fallosEsp=fallosEsp+1;
         else
             
         end
     end
 end
 
 especificidad=(aciertosEsp/(aciertosEsp+fallosEsp))*100;
 
 %Probabilidad siendo sano ir a sospechoso
  aciertosSanoasos=0;
 fallosSanoasos=0;
 for i=1:ms
     if TEtag(i,1)==1
         if TEtag_pred(i,1)==1
             aciertosSanoasos=aciertosSanoasos+1;
         elseif TEtag_pred(i,1)==2
             fallosSanoasos=fallosSanoasos+1;
         else
             
         end
     end
 end
 
 probDeSerSanoEIrASospechoso=(fallosSanoasos/(fallosSanoasos+aciertosSanoasos))*100;
 
 aciertosEnfasos=0;
 fallosEnfasos=0;
 for i=1:ms
     if TEtag(i,1)==3
         if TEtag_pred(i,1)==3
             aciertosEnfasos=aciertosEnfasos+1;
         elseif TEtag_pred(i,1)==2
             fallosEnfasos=fallosEnfasos+1;
         else
             
         end
     end
 end
 
 probDeEstarEnfermoEIrASospechoso=(fallosEnfasos/(fallosEnfasos+aciertosEnfasos))*100;
 
 
  aciertossosasos=0;
 fallossosasano=0;
 for i=1:ms
     if TEtag(i,1)==2
         if TEtag_pred(i,1)==2
             aciertossosasos=aciertossosasos+1;
         elseif TEtag_pred(i,1)==1
             fallossosasano=fallossosasano+1;
         else
             
         end
     end
 end
 
 probDeSerSospechosoEIrASano=(fallossosasano/(fallossosasano+aciertossosasos))*100;
 
 
   aciertossosasos=0;
 fallossosaenf=0;
 for i=1:ms
     if TEtag(i,1)==2
         if TEtag_pred(i,1)==2
             aciertossosasos=aciertossosasos+1;
         elseif TEtag_pred(i,1)==3
             fallossosaenf=fallossosaenf+1;
         else
             
         end
     end
 end
 
 probDeSerSospechosoEIrAEnf=(fallossosaenf/(fallossosaenf+aciertossosasos))*100;
