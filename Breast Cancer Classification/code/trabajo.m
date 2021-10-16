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

[m1,~]=size(mat1);
[m2,n2]=size(mat2);
[m3,n3]=size(mat3);



% for i=1:1:21
%     var = CORPUS_TRAIN(:,i);
%     var1 = mat1(:,i);
%     var2= mat2(:,i);
%     var3=mat3(:,i);
%     [f,xi] = ksdensity(var);
%     figure;
%     
%     subplot(2,2,1)
%     boxplot(var,CORPUS_TRAIN(:,n));
%     title(['boxplot ', etiquetas(i)]);
%     
%     subplot(2,2,2)
%     hist(var1);
%     hold on
%     hist(var2);
%     hold on
%     hist(var3);
%     title(['Histograma ', etiquetas(i)]);
%     
%     
%     subplot(2,2,3)
%     plot(xi,f);
%     title(['Densidad k ', etiquetas(i)]);
%     
%     subplot(2,2,4)
%     qqplot(var,CORPUS_TRAIN(:,n));
%     title(['QQPlot ', etiquetas(i)]);
%     
%     
% end

%Gr�fico de dispersion de cada clase

% plotmatrix(mat1(:,1:n-1),mat1(:,1:n-1));
% title('matriz de dispersion clase 1');
% figure;
% plotmatrix(mat2(:,1:n-1),mat2(:,1:n-1));
% title('matriz de dispersion clase 2');
% figure;
% plotmatrix(mat3(:,1:n-1),mat3(:,1:n-1));
% title('matriz de dispersion clase 3');
% 
%Matriz de correlaci�n de cada clase
% matrizCorr= corr(CORPUS_TRAIN(:,1:n-1), CORPUS_TRAIN(:,1:n-1));
% matrizCorr1 = corr(mat1(:,1:n-1), mat1(:,1:n-1));
% matrizCorr2 = corr(mat2(:,1:n-1), mat2(:,1:n-1));
% matrizCorr3 = corr(mat3(:,1:n-1), mat3(:,1:n-1));


 
outliers1 = isoutlier(mat1);
outliers2 = isoutlier(mat2);
outliers3 = isoutlier(mat3);
 

 
outliers = {outliers1,outliers2,outliers3}; %Celda para guardar las matrices
 
%Para clase 1:
cont=0; %Inicializo un contador para evitar que al perder filas de la matriz
       %se me generen errores
    
for i=1:length(outliers1)    
    if sum(int8(outliers1(i,:)))>4%Si el N� de outliers es mayor que la    
                                  %condici�n...
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

%Aplicamos lo anterior a la matriz de Validacion
MV = ismissing(CORPUS_TEST_table);
[M, N] = size(MV);
for i=1:M
    for j=1:(N-1)
        if(MV(i, j) == 1)
            CORPUS_TEST(i, j) = nanmean(CORPUS_TEST(:, j));
        end
    end
end

MV = ismissing(CORPUS_TEST);
etiquetas = CORPUS_TEST_table.Properties.VariableNames;


[m,n]=size(CORPUS_TEST);

ac1=0;
ac2=0;


for i=1:1:m
    if CORPUS_TEST(i,n)==1
        mat1t(i,:)=CORPUS_TEST(i,:);
        ac1=ac1+1;
    elseif CORPUS_TEST(i,n)==2
        mat2t(i-ac1,:)=CORPUS_TEST(i,:);
        ac2=ac2+1;
    else
        mat3t(i-(ac2+ac1),:)=CORPUS_TEST(i,:);
        
    end
end

[m1,n1]=size(mat1t);
[m2,n2]=size(mat2t);
[m3,n3]=size(mat3t);



% for i=1:1:21
%     var = CORPUS_TEST(:,i);
%     var1 = mat1(:,i);
%     var2= mat2(:,i);
%     var3=mat3(:,i);
%     [f,xi] = ksdensity(var);
%     figure;
%     
%     subplot(2,2,1)
%     boxplot(var,CORPUS_TEST(:,n));
% %     title(['boxplot ', etiquetas(i)]);
% %     
% %     subplot(2,2,2)
% %     hist(var1);
% %     hold on
% %     hist(var2);
% %     hold on
% %     hist(var3);
% %     title(['Histograma ', etiquetas(i)]);
% %     
% %     
% %     subplot(2,2,3)
% %     plot(xi,f);
% %     title(['Densidad k ', etiquetas(i)]);
% %     
% %     subplot(2,2,4)
% %     qqplot(var,CORPUS_TEST(:,n));
% %     title(['QQPlot ', etiquetas(i)]);
% %     
% %     
% % end
% 
% %Gr�fico de dispersion de cada clase
% 
% % plotmatrix(mat1(:,1:n-1),mat1(:,1:n-1));
% % title('matriz de dispersion clase 1');
% % figure;
% % plotmatrix(mat2(:,1:n-1),mat2(:,1:n-1));
% % title('matriz de dispersion clase 2');
% % figure;
% % plotmatrix(mat3(:,1:n-1),mat3(:,1:n-1));
% % title('matriz de dispersion clase 3');
% % 
% %Matriz de correlaci�n de cada clase
% % matrizCorr= corr(CORPUS_TEST(:,1:n-1), CORPUS_TEST(:,1:n-1));
% % matrizCorr1 = corr(mat1(:,1:n-1), mat1(:,1:n-1));
% % matrizCorr2 = corr(mat2(:,1:n-1), mat2(:,1:n-1));
% % matrizCorr3 = corr(mat3(:,1:n-1), mat3(:,1:n-1));
% 
% 
%  
outliers1 = isoutlier(mat1t);
outliers2 = isoutlier(mat2t);
outliers3 = isoutlier(mat3t);

outliers = {outliers1,outliers2,outliers3}; %Celda para guardar las matrices
 
%Para clase 1:
cont=0; %Inicializo un contador para evitar que al perder filas de la matriz
       %se me generen errores
    
for i=1:length(outliers1)    
    if sum(int8(outliers1(i,:)))>4%Si el N� de outliers es mayor que la    
                                  %condici�n...
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
[m,n]=size(CORPUS_TEST);


    
    

%4.2
%filters
[out, rank] = fscore(CORPUS_TRAIN(:,1:n-1),CORPUS_TRAIN(:,n)); 
[ranks,weights] = relieff(CORPUS_TRAIN(:,1:n-1),CORPUS_TRAIN(:,n),m); 

%Sacamos las columnas mas significativas
%fscore
% columna1f=rank(1,1);
% columna2f=rank(2,1);
% 
% scatter(CORPUS_TRAIN(1:m1,columna1f),CORPUS_TRAIN(1:m1,columna2f),'b*');
% hold on;
% scatter(CORPUS_TRAIN(m1:m2+m1,columna1f),CORPUS_TRAIN(m1:m2+m1,columna2f),'r*');
% hold on;
% scatter(CORPUS_TRAIN(m2+m1:m3+m1+m2,columna1f),CORPUS_TRAIN(m2+m1:m3+m1+m2,columna2f),'k*');
% title('Fscore');
% hold off;
% figure;

%relieff
% columna1r=ranks(1,1);
% columna2r=ranks(1,2);

% scatter(CORPUS_TRAIN(1:m1,columna1r),CORPUS_TRAIN(1:m1,columna2r),'b*');
% hold on;
% scatter(CORPUS_TRAIN(m1:m2+m1,columna1r),CORPUS_TRAIN(m1:m2+m1,columna2r),'r*');
% hold on;
% scatter(CORPUS_TRAIN(m2+m1:m3+m1+m2,columna1r),CORPUS_TRAIN(m2+m1:m3+m1+m2,columna2r),'k*');
% title('Relieff');
% hold off;
% figure;
% %wrapers
% inmodel = sequentialfs(fun,CORPUS_TRAIN(:,1:n-1),CORPUS_TRAIN(:,n));
% inmodel2 = sequentialfs(fun,CORPUS_TRAIN(:,1:n-1),CORPUS_TRAIN(:,n),'direction','backward');

%PCA

[coeffpca,scorepca,latentpca] = pca(CORPUS_TRAIN(:,1:n-1));

matrizNuevaPCA=CORPUS_TRAIN(:,1:n-1)*coeffpca;
matrizNuevaPCA(:,n)=CORPUS_TRAIN(:,n);


% pareto(latentpca);
% biplot(coeffpca(:,1:3),'Scores',scorepca(1:m1,1:3),'Color','r');
% hold on;
% biplot(coeffpca(:,1:3),'Scores',scorepca(m1:m1+m2,1:3),'Color','b');
% hold on;
% biplot(coeffpca(:,1:3),'Scores',scorepca(m1+m2:m1+m2+m3,1:3),'Color','k');
% hold off;




% scatter(scorepca(1:m1,1),scorepca(1:m1,2),'b*');
% hold on;
% scatter(scorepca(m1:m2+m1,1),scorepca(m1:m2+m1,2),'r*');
% hold on;
% scatter(scorepca(m2+m1:m3+m1+m2,1),scorepca(m2+m1:m3+m1+m2,2),'k*');
% title('PCA');
% hold off;

%Aplicamos todo lo de antes con el Z score
Z = zscore(CORPUS_TRAIN(:,1:n-1));
Z(:,n)=CORPUS_TRAIN(:,n);

%aplicamos sobre la matriz de test
Zt = zscore(CORPUS_TEST(:,1:n-1));
Zt(:,n)=CORPUS_TEST(:,n);
% % 
[coeffz,scorez,latentz] = pca(Z(:,1:n-1));
% pareto(latentz);

[coeffzt,scorezt,latentzt] = pca(Zt(:,1:n-1));
% pareto(latentzt);
% % % 
% biplot(coeffz(:,1:3),'Scores',scorez(1:m1,1:3),'Color','r');
% hold on;
% biplot(coeffz(:,1:3),'Scores',scorez(m1:m1+m2,1:3),'Color','b');
% hold on;
% biplot(coeffz(:,1:3),'Scores',scorez(m1+m2:m1+m2+m3,1:3),'Color','k');
% hold off;
% % 
% scatter(scorez(1:m1,1),scorez(1:m1,2),'b*');
% hold on;
% scatter(scorez(m1:m2+m1,1),scorez(m1:m2+m1,2),'r*');
% hold on;
% scatter(scorez(m2+m1:m3+m1+m2,1),scorez(m2+m1:m3+m1+m2,2),'k*');
% title('PCA Z');
% hold off;
% 
[outZ, rankZ] = fscore(Z(:,1:n-1),Z(:,n)); 
[ranksZ,weightsZ] = relieff(Z(:,1:n-1),Z(:,n),m); 

[outZt, rankZt] = fscore(Zt(:,1:n-1),Zt(:,n)); 
[ranksZt,weightsZt] = relieff(Zt(:,1:n-1),Zt(:,n),m); 
% 
% figure;
% 
% columna1f=rankZ(1,1);
% columna2f=rankZ(2,1);
% 
% scatter(Z(1:m1,columna1f),Z(1:m1,columna2f),'b*');
% hold on;
% scatter(Z(m1:m2+m1,columna1f),Z(m1:m2+m1,columna2f),'r*');
% hold on;
% scatter(Z(m2+m1:m3+m1+m2,columna1f),Z(m2+m1:m3+m1+m2,columna2f),'k*');
% title('Fscore Z');
% hold off;
% figure;
% 
% 
% columna1r=ranksZ(1,1);
% columna2r=ranksZ(1,2);
% 
% scatter(Z(1:m1,columna1r),Z(1:m1,columna2r),'b*');
% hold on;
% scatter(Z(m1:m2+m1,columna1r),Z(m1:m2+m1,columna2r),'r*');
% hold on;
% scatter(Z(m2+m1:m3+m1+m2,columna1r),Z(m2+m1:m3+m1+m2,columna2r),'k*');
% title('Relieff Z');
% hold off;



[eigenmodel, W, l, ProjectedData] = flda(CORPUS_TRAIN(:,1:n-1), CORPUS_TRAIN(:,n) );
matdatos = CORPUS_TRAIN(:,1:n-1)*W;
matdatos(:,n)=CORPUS_TRAIN(:,n);

[eigenmodelt, Wt, lt, ProjectedDatat] = flda(CORPUS_TEST(:,1:n-1), CORPUS_TEST(:,n) );
matdatost = CORPUS_TEST(:,1:n-1)*Wt;
matdatost(:,n)=CORPUS_TEST(:,n);
% 
% biplot(W(:,1:2), 'score', matdatos(1:m1, 1:2), 'color', 'g');
% hold on;
% biplot(W(:,1:2), 'score', matdatos(m1:m1+m2, 1:2), 'color', 'b');
% hold on;
% biplot(W(:,1:2), 'score', matdatos(m1+m2:m1+m2+m3, 1:2), 'color', 'k');
% hold off;


[eigenmodelz, Wz, lz, ProjectedDataz] = flda(Z(:,1:n-1), Z(:,n) );
matdatosz = Z(:,1:n-1)*Wz;
matdatosz(:,n)=Z(:,n);

[eigenmodelzt, Wzt, lzt, ProjectedDatazt] = flda(Zt(:,1:n-1), Zt(:,n) );
matdatoszt = Zt(:,1:n-1)*Wzt;
matdatoszt(:,n)=Zt(:,n);
% 
% biplot(Wz(:,1:2), 'score', matdatosz(1:m1, 1:2), 'color', 'g');
% hold on;
% biplot(Wz(:,1:2), 'score', matdatosz(m1:m1+m2, 1:2), 'color', 'b');
% hold on;
% biplot(Wz(:,1:2), 'score', matdatosz(m1+m2:m1+m2+m3, 1:2), 'color', 'k');
% hold off;






















todoError= zeros(1,32);
tipoError = strings(1,32);





%CORPUS_TRAIN

%PCA

erroresLineal= zeros(1,100);
erroresCuadratico= zeros(1,100);
erroresKV1= zeros(1,100);
erroresKV50= zeros(1,100);
for i=1:1:100
%     CV=cvpartition(m,'HoldOut',0.3);
% 
% % Separamos los conjuntos de entrenamiento y validación
% TRdata=CORPUS_TRAIN(CV.training,1:end-1);
% TRtag=CORPUS_TRAIN(CV.training,end);
% TEdata=CORPUS_TRAIN(CV.test,1:end-1);
% TEtag=CORPUS_TRAIN(CV.test,end);


%Usamos PCA sobre el conjunto de entrenamiento para extraer 
%características mas representativas.

[coef,TRscore]=pca(CORPUS_TRAIN(:,1:end-1));
% [coef,TEscore]=pca(CORPUS_TEST(:,1:end-1));

TEscore = (CORPUS_TEST(:,1:end-1)-mean(CORPUS_TRAIN(:,1:end-1)))*coef;
%Aplicamos la transformación PCA obtenida del conjunto de entrenamiento
%sobre el conjunto de validación


%Nos quedamos con los 7 componentes principales como características
numPC=5;
TRdata=TRscore(:,1:numPC);
TEdata=TEscore(:,1:numPC);

[me,ne] = size(TRdata);
[mt,nt] = size(TEdata);
TEtag=CORPUS_TEST(:,end);
TRtag=CORPUS_TRAIN(:,end);
%LINEAL
    LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','linear');
    TEtag_pred=predict(LinModel,TEdata);
    Error_lineal=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresLineal(1,i)=Error_lineal;
%CUADRATICO
LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','quadratic');
    TEtag_pred=predict(LinModel,TEdata);
    Error_cuadra=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresCuadratico(1,i)=Error_cuadra;
 
%K-VECINOS 1
 mdvec1 = fitcknn(TRdata,TRtag, 'NumNeighbors',1);
 TEtag_pred = predict(mdvec1, TEdata);
 Error_K1=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos1(1,i)=Error_K1;
 
 %K-VECINOS 50
 mdvec50 = fitcknn(TRdata,TRtag, 'NumNeighbors',50);
 TEtag_pred = predict(mdvec50, TEdata);
 Error_K50=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos50(1,i)=Error_K50;
end

%LINEAL
mediaErrorLinealPCA=mean(erroresLineal);
desviacionErrorLinealPCA=std(erroresLineal);
todoError(1,1)=mediaErrorLinealPCA;
tipoError(1,1)='mediaErrorLinealPCA';
%CUADRATICO
mediaErrorCuadraticoPCA=mean(erroresCuadratico);
desviacionErrorCuadraticoPCA=std(erroresCuadratico);
todoError(1,2)=mediaErrorCuadraticoPCA;
tipoError(1,2)='mediaErrorCuadraticoPCA';
%KV 1
mediaErrorKVecinos1PCA=mean(erroresKVecinos1);
desviacionErrorKVecinos1PCA=std(erroresKVecinos1);
todoError(1,3)=mediaErrorKVecinos1PCA;
tipoError(1,3)='desviacionErrorKVecinos1PCA';
%KV 50
mediaErrorKVecinos50PCA=mean(erroresKVecinos50);
desviacionErrorKVecinos50PCA=std(erroresKVecinos50);
todoError(1,4)=mediaErrorKVecinos50PCA;
tipoError(1,4)='mediaErrorKVecinos50PCA';




%RELIEFF

erroresLineal= zeros(1,100);
erroresCuadratico= zeros(1,100);
erroresKV1= zeros(1,100);
erroresKV50= zeros(1,100);
for i=1:1:100
%     CV=cvpartition(m,'HoldOut',0.3);
% 
% % Separamos los conjuntos de entrenamiento y validación
% TRdata=CORPUS_TRAIN(CV.training,1:end-1);
% TRtag=CORPUS_TRAIN(CV.training,end);
% TEdata=CORPUS_TRAIN(CV.test,1:end-1);
% TEtag=CORPUS_TRAIN(CV.test,end);

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

%LINEAL
    LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','linear');
    TEtag_pred=predict(LinModel,TEdata);
    Error_lineal=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresLineal(1,i)=Error_lineal;
%CUADRATICO

LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','quadratic');
    TEtag_pred=predict(LinModel,TEdata);
    Error_cuadra=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresCuadratico(1,i)=Error_cuadra;
 
%K-VECINOS 1
 mdvec1 = fitcknn(TRdata,TRtag, 'NumNeighbors',1);
 TEtag_pred = predict(mdvec1, TEdata);
 Error_K1=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos1(1,i)=Error_K1;
 
 %K-VECINOS 50
 mdvec50 = fitcknn(TRdata,TRtag, 'NumNeighbors',50);
 TEtag_pred = predict(mdvec50, TEdata);
 Error_K50=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos50(1,i)=Error_K50;
end

%LINEAL
mediaErrorLinealRelf=mean(erroresLineal);
desviacionErrorLinealRelf=std(erroresLineal);
todoError(1,5)=mediaErrorLinealRelf;
tipoError(1,5)='mediaErrorLinealRelf';
%CUADRATICO
mediaErrorCuadraticoRelf=mean(erroresCuadratico);
desviacionErrorCuadraticoRelf=std(erroresCuadratico);
todoError(1,6)=mediaErrorCuadraticoRelf;
tipoError(1,6)='mediaErrorCuadraticoRelf';
%KV 1
mediaErrorKVecinos1Relf=mean(erroresKVecinos1);
desviacionErrorKVecinos1Relf=std(erroresKVecinos1);
todoError(1,7)=mediaErrorKVecinos1Relf;
tipoError(1,7)='mediaErrorKVecinos1Relf';
%KV 50
mediaErrorKVecinos50Relf=mean(erroresKVecinos50);
desviacionErrorKVecinos50Relf=std(erroresKVecinos50);
todoError(1,8)=mediaErrorKVecinos50Relf;
tipoError(1,8)='mediaErrorKVecinos50Relf';

%F-SCORE

erroresLineal= zeros(1,100);
erroresCuadratico= zeros(1,100);
erroresKV1= zeros(1,100);
erroresKV50= zeros(1,100);

for i=1:1:100
%     CV=cvpartition(m,'HoldOut',0.3);
% 
% % Separamos los conjuntos de entrenamiento y validación
% TRdata=CORPUS_TRAIN(CV.training,1:end-1);
% TRtag=CORPUS_TRAIN(CV.training,end);
% TEdata=CORPUS_TRAIN(CV.test,1:end-1);
% TEtag=CORPUS_TRAIN(CV.test,end);

seleccion2=10;
seleccion22=10;
for j=1:seleccion2
    matBienTrain(:,j)=CORPUS_TRAIN(:,rank(j,1));
end

for h=1:seleccion22
    matBienTest(:,h)=CORPUS_TEST(:,rank(h,1));
end

TRdata=matBienTrain;
TEdata=matBienTest;
TEtag=CORPUS_TEST(:,end);
TRtag=CORPUS_TRAIN(:,end);

[me,ne] = size(TRdata);
[mt,nt] = size(TEdata);

%LINEAL
    LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','linear');
    TEtag_pred=predict(LinModel,TEdata);
    Error_lineal=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresLineal(1,i)=Error_lineal;
%CUADRATICO
LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','quadratic');
    TEtag_pred=predict(LinModel,TEdata);
    Error_cuadra=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresCuadratico(1,i)=Error_cuadra;
 
%K-VECINOS 1
 mdvec1 = fitcknn(TRdata,TRtag, 'NumNeighbors',1);
 TEtag_pred = predict(mdvec1, TEdata);
 Error_K1=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos1(1,i)=Error_K1;
 
 %K-VECINOS 50
 mdvec50 = fitcknn(TRdata,TRtag, 'NumNeighbors',50);
 TEtag_pred = predict(mdvec50, TEdata);
 Error_K50=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos50(1,i)=Error_K50;
end

%LINEAL
mediaErrorLinealFSC=mean(erroresLineal);
desviacionErrorLinealFSC=std(erroresLineal);
todoError(1,9)=mediaErrorLinealFSC;
tipoError(1,9)='mediaErrorLinealFSC';
%CUADRATICO
mediaErrorCuadraticoFSC=mean(erroresCuadratico);
desviacionErrorCuadraticoFSC=std(erroresCuadratico);
todoError(1,10)=mediaErrorCuadraticoFSC;
tipoError(1,10)='mediaErrorCuadraticoFSC';
%KV 1
mediaErrorKVecinos1FSC=mean(erroresKVecinos1);
desviacionErrorKVecinos1FSC=std(erroresKVecinos1);
todoError(1,11)=mediaErrorKVecinos1FSC;
tipoError(1,11)='mediaErrorKVecinos1FSC';
%KV 50
mediaErrorKVecinos50FSC=mean(erroresKVecinos50);
desviacionErrorKVecinos50FSC=std(erroresKVecinos50);
todoError(1,12)=mediaErrorKVecinos50FSC;
tipoError(1,12)='mediaErrorKVecinos50FSC';

%LDA

erroresLineal= zeros(1,100);
erroresCuadratico= zeros(1,100);
erroresKV1= zeros(1,100);
erroresKV50= zeros(1,100);
for i=1:1:100
%     CV=cvpartition(m,'HoldOut',0.3);
numLDA=8;

    
    
% Separamos los conjuntos de entrenamiento y validación
% TRdata=matdatos(CV.training,1:end-1);
% TRtag=matdatos(CV.training,end);
% TEdata=matdatos(CV.test,1:end-1);
% TEtag=matdatos(CV.test,end);


TRdata=matdatos(:,1:numLDA);
TEdata=matdatost(:,1:numLDA);




[me,ne] = size(TRdata);
[mt,nt] = size(TEdata);
TEtag=matdatost(:,end);
TRtag=matdatos(:,end);
%LINEAL
    LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','linear');
    TEtag_pred=predict(LinModel,TEdata);
    Error_lineal=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresLineal(1,i)=Error_lineal;
%CUADRATICO
LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','quadratic');
    TEtag_pred=predict(LinModel,TEdata);
    Error_cuadra=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresCuadratico(1,i)=Error_cuadra;
 
%K-VECINOS 1
 mdvec1 = fitcknn(TRdata,TRtag, 'NumNeighbors',1);
 TEtag_pred = predict(mdvec1, TEdata);
 Error_K1=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos1(1,i)=Error_K1;
 
 %K-VECINOS 50
 mdvec50 = fitcknn(TRdata,TRtag, 'NumNeighbors',50);
 TEtag_pred = predict(mdvec50, TEdata);
 Error_K50=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos50(1,i)=Error_K50;
end

%LINEAL
mediaErrorLinealLDA=mean(erroresLineal);
desviacionErrorLinealLDA=std(erroresLineal);
todoError(1,13)=mediaErrorLinealLDA;
tipoError(1,13)='mediaErrorLinealLDA';
%CUADRATICO
mediaErrorCuadraticoLDA=mean(erroresCuadratico);
desviacionErrorCuadraticoLDA=std(erroresCuadratico);
todoError(1,14)=mediaErrorCuadraticoLDA;
tipoError(1,14)='mediaErrorCuadraticoLDA';
%KV 1
mediaErrorKVecinos1LDA=mean(erroresKVecinos1);
desviacionErrorKVecinos1LDA=std(erroresKVecinos1);
todoError(1,15)=mediaErrorKVecinos1LDA;
tipoError(1,15)='mediaErrorKVecinos1LDA';
%KV 50
mediaErrorKVecinos50LDA=mean(erroresKVecinos50);
desviacionErrorKVecinos50LDA=std(erroresKVecinos50);
todoError(1,16)=mediaErrorKVecinos50LDA;
tipoError(1,16)='mediaErrorKVecinos50LDA';


%Z

% CORPUS_TRAIN = Z;

%PCA

erroresLineal= zeros(1,100);
erroresCuadratico= zeros(1,100);
erroresKV1= zeros(1,100);
erroresKV50= zeros(1,100);
for i=1:1:100
%     CV=cvpartition(m,'HoldOut',0.3);
% 
% % Separamos los conjuntos de entrenamiento y validación
% TRdata=CORPUS_TRAIN(CV.training,1:end-1);
% TRtag=CORPUS_TRAIN(CV.training,end);
% TEdata=CORPUS_TRAIN(CV.test,1:end-1);
% TEtag=CORPUS_TRAIN(CV.test,end);


%Usamos PCA sobre el conjunto de entrenamiento para extraer 
%características mas representativas.

% [coef,TRscore]=pca(Z);

%Aplicamos la transformación PCA obtenida del conjunto de entrenamiento
%sobre el conjunto de validación
% TEscore=(TEdata-mean(TRdata))*coef;
TEscore = (Zt(:,1:end-1)-mean(Z(:,1:end-1)))*coeffz;
%Nos quedamos con los 7 componentes principales como características
numPC=10;
TRdata=scorez(:,1:numPC);
TEdata=scorezt(:,1:numPC);

[me,ne] = size(TRdata);
[mt,nt] = size(TEdata);
TEtag=Zt(:,end);
TRtag=Z(:,end);
%LINEAL
    LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','linear');
    TEtag_pred=predict(LinModel,TEdata);
    Error_lineal=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresLineal(1,i)=Error_lineal;
%CUADRATICO
LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','quadratic');
    TEtag_pred=predict(LinModel,TEdata);
    Error_cuadra=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresCuadratico(1,i)=Error_cuadra;
 
%K-VECINOS 1
 mdvec1 = fitcknn(TRdata,TRtag, 'NumNeighbors',1);
 TEtag_pred = predict(mdvec1, TEdata);
 Error_K1=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos1(1,i)=Error_K1;
 
 %K-VECINOS 50
 mdvec50 = fitcknn(TRdata,TRtag, 'NumNeighbors',50);
 TEtag_pred = predict(mdvec50, TEdata);
 Error_K50=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos50(1,i)=Error_K50;
end

%LINEAL
mediaErrorLinealZPCA=mean(erroresLineal);
desviacionErrorLinealZPCA=std(erroresLineal);
todoError(1,17)=mediaErrorLinealZPCA;
tipoError(1,17)='mediaErrorLinealZPCA';
%CUADRATICO
mediaErrorCuadraticoZPCA=mean(erroresCuadratico);
desviacionErrorCuadraticoZ=std(erroresCuadratico);
todoError(1,18)=mediaErrorCuadraticoZPCA;
tipoError(1,18)='mediaErrorCuadraticoZPCA';
%KV 1
mediaErrorKVecinos1ZPCA=mean(erroresKVecinos1);
desviacionErrorKVecinos1ZPCA=std(erroresKVecinos1);
todoError(1,19)=mediaErrorKVecinos1ZPCA;
tipoError(1,19)='mediaErrorKVecinos1ZPCA';
%KV 50
mediaErrorKVecinos50ZPCA=mean(erroresKVecinos50);
desviacionErrorKVecinos50ZPCA=std(erroresKVecinos50);
todoError(1,20)=mediaErrorKVecinos50ZPCA;
tipoError(1,20)='mediaErrorKVecinos50ZPCA';




%RELIEFF

erroresLineal= zeros(1,100);
erroresCuadratico= zeros(1,100);
erroresKVecinos1= zeros(1,100);
erroresKV50= zeros(1,100);
for i=1:1:100
%     CV=cvpartition(m,'HoldOut',0.3);
% 
% % Separamos los conjuntos de entrenamiento y validación
% TRdata=CORPUS_TRAIN(CV.training,1:end-1);
% TRtag=CORPUS_TRAIN(CV.training,end);
% TEdata=CORPUS_TRAIN(CV.test,1:end-1);
% TEtag=CORPUS_TRAIN(CV.test,end);

seleccion2=7;
seleccion22=7;
for v=1:seleccion2
    matBienTrain1(:,v)=Z(:,ranksZ(1,v));
end

for w=1:seleccion22
    matBienTest1(:,w)=Zt(:,ranksZ(1,w));
end

TRdata=matBienTrain1;
TEdata=matBienTest1;

[me,ne] = size(TRdata);
[mt,nt] = size(TEdata);
TEtag=Zt(:,end);
TRtag=Z(:,end);

%LINEAL
    LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','linear');
    TEtag_pred=predict(LinModel,TEdata);
    Error_lineal=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresLineal(1,i)=Error_lineal;
%CUADRATICO
LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','quadratic');
    TEtag_pred=predict(LinModel,TEdata);
    Error_cuadra=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresCuadratico(1,i)=Error_cuadra;
 
%K-VECINOS 1
 mdvec1 = fitcknn(TRdata,TRtag, 'NumNeighbors',1);
 TEtag_pred = predict(mdvec1, TEdata);
 Error_K1=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos1(1,i)=Error_K1;
 
 %K-VECINOS 50
 mdvec50 = fitcknn(TRdata,TRtag, 'NumNeighbors',50);
 TEtag_pred = predict(mdvec50, TEdata);
 Error_K50=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos50(1,i)=Error_K50;
end

%LINEAL
mediaErrorLinealRelfZ=mean(erroresLineal);
desviacionErrorLinealRelfZ=std(erroresLineal);
todoError(1,21)=mediaErrorLinealRelfZ;
tipoError(1,21)='mediaErrorLinealRelfZ';
%CUADRATICO
mediaErrorCuadraticoRelfZ=mean(erroresCuadratico);
desviacionErrorCuadraticoRelfZ=std(erroresCuadratico);
todoError(1,22)=mediaErrorCuadraticoRelfZ;
tipoError(1,22)='mediaErrorCuadraticoRelfZ';
%KV 1
mediaErrorKVecinos1RelfZ=mean(erroresKVecinos1);
desviacionErrorKVecinos1RelfZ=std(erroresKVecinos1);
todoError(1,23)=mediaErrorKVecinos1RelfZ;
tipoError(1,23)='mediaErrorKVecinos1RelfZ';
%KV 50
mediaErrorKVecinos50RelfZ=mean(erroresKVecinos50);
desviacionErrorKVecinos50RelfZ=std(erroresKVecinos50);
todoError(1,24)=mediaErrorKVecinos50RelfZ;
tipoError(1,24)='mediaErrorKVecinos50RelfZ';

%F-SCORE

erroresLineal= zeros(1,100);
erroresCuadratico= zeros(1,100);
erroresKV1= zeros(1,100);
erroresKV50= zeros(1,100);
for i=1:1:100
%     CV=cvpartition(m,'HoldOut',0.3);
% 
% % Separamos los conjuntos de entrenamiento y validación
% TRdata=CORPUS_TRAIN(CV.training,1:end-1);
% TRtag=CORPUS_TRAIN(CV.training,end);
% TEdata=CORPUS_TRAIN(CV.test,1:end-1);
% TEtag=CORPUS_TRAIN(CV.test,end);

seleccion2=10;
seleccion22=10;
for j=1:seleccion2
    matBienTrain3(:,j)=Z(:,rankZ(j,1));
end

for h=1:seleccion22
    matBienTest3(:,h)=Zt(:,rankZ(h,1));
end
TRdata=matBienTrain3;
TEdata=matBienTest3;
TEtag=Zt(:,end);
TRtag=Z(:,end);

[me,ne] = size(TRdata);
[mt,nt] = size(TEdata);

%LINEAL
    LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','linear');
    TEtag_pred=predict(LinModel,TEdata);
    Error_lineal=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresLineal(1,i)=Error_lineal;
%CUADRATICO
LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','quadratic');
    TEtag_pred=predict(LinModel,TEdata);
    Error_cuadra=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresCuadratico(1,i)=Error_cuadra;
 
%K-VECINOS 1
 mdvec1 = fitcknn(TRdata,TRtag, 'NumNeighbors',1);
 TEtag_pred = predict(mdvec1, TEdata);
 Error_K1=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos1(1,i)=Error_K1;
 
 %K-VECINOS 50
 mdvec50 = fitcknn(TRdata,TRtag, 'NumNeighbors',50);
 TEtag_pred = predict(mdvec50, TEdata);
 Error_K50=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos50(1,i)=Error_K50;
end

%LINEAL
mediaErrorLinealFSCZ=mean(erroresLineal);
desviacionErrorLinealFSCZ=std(erroresLineal);
todoError(1,25)=mediaErrorLinealFSCZ;
tipoError(1,25)='mediaErrorLinealFSCZ';
%CUADRATICO
mediaErrorCuadraticoFSCZ=mean(erroresCuadratico);
desviacionErrorCuadraticoFSCZ=std(erroresCuadratico);
todoError(1,26)=mediaErrorCuadraticoFSCZ;
tipoError(1,26)='mediaErrorCuadraticoFSCZ';
%KV 1
mediaErrorKVecinos1FSCZ=mean(erroresKVecinos1);
desviacionErrorKVecinos1FSCZ=std(erroresKVecinos1);
todoError(1,27)=mediaErrorKVecinos1FSCZ;
tipoError(1,27)='mediaErrorKVecinos1FSCZ';
%KV 50
mediaErrorKVecinos50FSCZ=mean(erroresKVecinos50);
desviacionErrorKVecinos50FSCZ=std(erroresKVecinos50);
todoError(1,28)=mediaErrorKVecinos50FSCZ;
tipoError(1,28)='mediaErrorKVecinos50FSCZ';

%LDA

erroresLineal= zeros(1,100);
erroresCuadratico= zeros(1,100);
erroresKV1= zeros(1,100);
erroresKV50= zeros(1,100);
for i=1:1:100
%     CV=cvpartition(m,'HoldOut',0.3);
numLDA1=8;

    
    
% Separamos los conjuntos de entrenamiento y validación
% TRdata=matdatosz(CV.training,1:end-1);
% TRtag=matdatosz(CV.training,end);
% TEdata=matdatosz(CV.test,1:end-1);
% TEtag=matdatosz(CV.test,end);

TRdata=matdatosz(:,1:numLDA1);
TEdata=matdatoszt(:,1:numLDA1);




[me,ne] = size(TRdata);
[mt,nt] = size(TEdata);
TEtag=Zt(:,end);
TRtag=Z(:,end);

%LINEAL
    LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','linear');
    TEtag_pred=predict(LinModel,TEdata);
    Error_lineal=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresLineal(1,i)=Error_lineal;
%CUADRATICO
LinModel=fitcdiscr(TRdata,TRtag,'DiscrimType','quadratic');
    TEtag_pred=predict(LinModel,TEdata);
    Error_cuadra=(sum(TEtag~=TEtag_pred)/length(TEtag));
    erroresCuadratico(1,i)=Error_cuadra;
 
%K-VECINOS 1
 mdvec1 = fitcknn(TRdata,TRtag, 'NumNeighbors',1);
 TEtag_pred = predict(mdvec1, TEdata);
 Error_K1=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos1(1,i)=Error_K1;
 
 %K-VECINOS 50
 mdvec50 = fitcknn(TRdata,TRtag, 'NumNeighbors',50);
 TEtag_pred = predict(mdvec50, TEdata);
 Error_K50=(sum(TEtag~=TEtag_pred)/length(TEtag));
 erroresKVecinos50(1,i)=Error_K50;
end

%LINEAL
mediaErrorLinealLDAZ=mean(erroresLineal);
desviacionErrorLinealLDAZ=std(erroresLineal);
todoError(1,29)=mediaErrorLinealLDAZ;
tipoError(1,29)='mediaErrorLinealLDAZ';
%CUADRATICO
mediaErrorCuadraticoLDAZ=mean(erroresCuadratico);
desviacionErrorCuadraticoLDAZ=std(erroresCuadratico);
todoError(1,30)=mediaErrorCuadraticoLDAZ;
tipoError(1,30)='mediaErrorCuadraticoLDAZ';
%KV 1
mediaErrorKVecinos1LDAZ=mean(erroresKVecinos1);
desviacionErrorKVecinos1LDAZ=std(erroresKVecinos1);
todoError(1,31)=mediaErrorKVecinos1LDAZ;
tipoError(1,31)='mediaErrorKVecinos1LDAZ';
%KV 50
mediaErrorKVecinos50LDAZ=mean(erroresKVecinos50);
desviacionErrorKVecinos50LDAZ=std(erroresKVecinos50);
todoError(1,32)=mediaErrorKVecinos50LDAZ;
tipoError(1,32)='mediaErrorKVecinos50LDAZ';


errorMin=min(todoError);
posicion = find(todoError==errorMin);
texto = tipoError(1,posicion);

str=(['El menor error corresponde a: ', errorMin*100, '%, este corresponde a el modelo ',texto]);


b = bar(todoError);
b.FaceColor = 'flat';
b.CData(posicion,:) = [.5 0 0];
title(str);
xlabel('tipos de Modelos de aprendizaje');
ylabel('Error medio cometido en tanto por 1');
