matriz = readtable('database_cancer.xlsx');

%Tranformamos la segunda columna (correspondiente a las clases) en 1 o 2
%1-Maligno
%2-Benigno

clases = matriz(:,2);
[m1,n1] = size(clases);
clases = table2array(clases);
for i=1:m1
    if clases{i}=='M'
        clases_new(i,1)=1;
    else
        clases_new(i,1)=2;
    end
end

%Limpio ademas la matriz general de las clases y la primera columna (ya que
%se trata solo de un identificador)

matriz(:,1)=[];
matriz(:,1)=[];

%Calculamos la matriz de autocorrelaciones entre todas las variables para
%ver si alguna variable esta muy correlacionada con otra y asi poder
%eliminarla. Para ello voy a realizar este estudio por clases, ya que es
%mucho mas eficiente, separando en dos matrices la matriz incial segun la
%etiqueta de estas

indexM=1;
indexB=1;

for j=1:m1
    if clases_new(j,1)==1
        matM(indexM,:)= matriz(j,:);
        indexM = indexM +1;
    else
        matB(indexB,:)=matriz(j,:);
        indexB = indexB +1;
    end
end

matriz = table2array(matriz);
matM = table2array(matM);
matB = table2array(matB);

%Correlaciones de ambas:

corrM = corr(matM, matM);
corrB = corr(matB, matB);

corrM=reshape(corrM(~diag(ones(1,size(corrM, 1)))), size(corrM)-[1 0]);
corrB=reshape(corrB(~diag(ones(1,size(corrB, 1)))), size(corrB)-[1 0]);

detM = corrM>0.8 | corrM<-0.8;
detB = corrB>0.8 | corrB<-0.8;



%eliminamos las variables que superen el umbral de correlacion

[m2,n2] = size(detM);

for i=1:m2
    for j=1:n2
        if detM(i,j)==1
            matriz(:,i)=[];
        end
        if detB(i,j)==1
            matriz(:,i)=[];
        end
    end
end

%Ya tenemos la matriz prefiltrada, es hora de ordenar sus variables por el
%orden de pesos que tengan, para ello aplicaremos diferentes metodos
%basados en el aprendizaje supervisado, es decir, conociendo el valor de
%las etiquetas de clase.
% 
% [m,n] = size(matriz);
% %fscore:
% [out, rank] = fscore(matriz,clases_new); 
% rank = rank';
% %relieff:
% [ranks,weights] = relieff(matriz,clases_new,m+1); 
% 
% %FISHER
% % a = fisher(matriz);
% 
% %MRMR
% 
% % [fea, score] = mRMR(matriz, clases_new, 10)
% 
% %UDFS
% % ranking = UDFS(matriz, clases_new);
% % ranking = ranking';
% 
% %mutInfFS
% [ rankmut , w] = mutInfFS( matriz,clases_new,10);
% 
% [out,idx] = sort(w,1); 
% sortedmat = rankmut(idx,:);
% rankmut = sortedmat;
% 
% %MCFS_p
% % [FeaIndex,FeaNumCandi] = MCFS_p(matriz,clases_new)
% 
% %LaPlacianScore
% % [Y] = LaplacianScore(matriz, clases_new);
% % [B,rankL] = sortrows(Y);
% % rankL = rankL';
% 
% %infFS
% % [rankInf, WEIGHTINF] = infFS( matriz, clases_new, 0.1, 0, 0 );
% 
% %ILFS
%  [rankILFS, WEIGHTILFS] = ILFS(matriz, clases_new  );
%  
% %  %fsvFS
%  [ rankfsvFS , w1] = fsvFS( matriz,clases_new,30 );
%  
%  rankfsvFS = rankfsvFS';
%  
% %FSASL
% % [W, S, A, objHistory] = FSASL(matriz, clases_new);
% 
% %ECFS
% % [ rankingECFS ] = ECFS( matriz, clases_new, 0.1 );
% 
% 
% 
% %Montamos la matriz de rankings (ordenados siendo la mayor la primera
% %columna y menos la ultima, cada fila es un metodo, el orden será el
% %siguiente):
% %1-fscore
% %2-relieff
% %3-ILFS
% %4-fsvFS
% 
% 
% mat_ranks =[rank; ranks; rankILFS; rankfsvFS];












