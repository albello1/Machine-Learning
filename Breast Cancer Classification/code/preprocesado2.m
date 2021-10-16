matriz = readtable('database_kris.xlsx');

%procedo a transformar los datos de esta base de datos a datos con los que
%podamos trabajar

%CLASES
%1-No recurrence
%2-Recurrence
clases = table2array(matriz(:,1));
clases = grp2idx(clases);

%AGE

%Estan del 1 al 9 ordenados por orden ascentente (como en el vector)
[m,n] = size(clases);
vector = ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"];

age = table2array(matriz(:,2));
age_new = zeros(m,1);

for i=1:m
    if age(i,1)==vector(1,1)
        age_new(i,1)=1;
    elseif age(i,1)==vector(1,2)
        age_new(i,1)=2;
    elseif age(i,1)==vector(1,3)
        age_new(i,1)=3;
    elseif age(i,1)==vector(1,4)
        age_new(i,1)=4;
    elseif age(i,1)==vector(1,5)
        age_new(i,1)=5;
    elseif age(i,1)==vector(1,6)
        age_new(i,1)=6;
    elseif age(i,1)==vector(1,7)
        age_new(i,1)=7;
    elseif age(i,1)==vector(1,8)
        age_new(i,1)=8;
    else
        age_new(i,1)=9;
    end
end

%MENOPAUSE

%1-premeo
%2-ge40
%3-it40

menopause = table2array(matriz(:,3));
manopause_num = grp2idx(menopause);

%lo transformamos en one-hot encoding
menopause_new = zeros(m,3);
for i=1:m
    if manopause_num(i,1)==1
        menopause_new(i,1)=1;
    elseif manopause_num(i,1)==2
        menopause_new(i,2)=1;
    else
        menopause_new(i,3)=1;
    end
end

        
        

%TUMOR-SIZE

%Ordenados en orden ascendente como en el vector

vector = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44","45-49", "50-54", "55-59"];

tumor = table2array(matriz(:,4));
tumor_new = zeros(m,1);

for i=1:m
    if tumor(i,1)==vector(1,1)
        tumor_new(i,1)=1;
    elseif tumor(i,1)==vector(1,2)
        tumor_new(i,1)=2;
    elseif tumor(i,1)==vector(1,3)
        tumor_new(i,1)=3;
    elseif tumor(i,1)==vector(1,4)
        tumor_new(i,1)=4;
    elseif tumor(i,1)==vector(1,5)
        tumor_new(i,1)=5;
    elseif tumor(i,1)==vector(1,6)
        tumor_new(i,1)=6;
    elseif tumor(i,1)==vector(1,7)
        tumor_new(i,1)=7;
    elseif tumor(i,1)==vector(1,8)
        tumor_new(i,1)=8;
    elseif tumor(i,1)==vector(1,9)
        tumor_new(i,1)=9;
    elseif tumor(i,1)==vector(1,10)
        tumor_new(i,1)=10;
    elseif tumor(i,1)==vector(1,11)
        tumor_new(i,1)=11;
    elseif tumor(i,1)==vector(1,12)
       tumor_new(i,1)=12;
    else
        tumor_new(i,1)=13;
        
    end
end

%INV-NODES

%Ordenados en orden ascendente como en el vector

vector =["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26","27-29", "30-32", "33-35", "36-39"]

nodes = table2array(matriz(:,5));
nodes_new = zeros(m,1);

for i=1:m
    if nodes(i,1)==vector(1,1)
        nodes_new(i,1)=1;
    elseif nodes(i,1)==vector(1,2)
        nodes_new(i,1)=2;
    elseif nodes(i,1)==vector(1,3)
        nodes_new(i,1)=3;
    elseif nodes(i,1)==vector(1,4)
        nodes_new(i,1)=4;
    elseif nodes(i,1)==vector(1,5)
        nodes_new(i,1)=5;
    elseif nodes(i,1)==vector(1,6)
        nodes_new(i,1)=6;
    elseif nodes(i,1)==vector(1,7)
        nodes_new(i,1)=7;
    elseif nodes(i,1)==vector(1,8)
        nodes_new(i,1)=8;
    elseif nodes(i,1)==vector(1,9)
        nodes_new(i,1)=9;
    elseif nodes(i,1)==vector(1,10)
        nodes_new(i,1)=10;
    elseif nodes(i,1)==vector(1,11)
        nodes_new(i,1)=11;
    elseif nodes(i,1)==vector(1,12)
       nodes_new(i,1)=12;
    else
        nodes_new(i,1)=13;
        
    end
end

%NODE CAPS

%1-no
%2-yes
%3-? (no se sabe)

node = table2array(matriz(:,6));
node = grp2idx(node);

node_new = zeros(m,3);
for i=1:m
    if node(i,1)==1
        node_new(i,1)=1;
    elseif node(i,1)==2
        node_new(i,2)=1;
    else
        node_new(i,3)=1;
    end
end



%BREAST

%1-left
%2-right

breast = table2array(matriz(:,8));
breast = grp2idx(breast);

%BREAST-SQUAD

%1-left-low
%2-right-up
%3-left-up
%4-right-low
%5-central

breastsqd = table2array(matriz(:,9));
breastsqd = grp2idx(breastsqd);

breastsqd_new = zeros(m,5);
for i=1:m
    if breastsqd(i,1)==1
        breastsqd_new(i,1)=1;
    elseif breastsqd(i,1)==2
        breastsqd_new(i,2)=1;
    elseif breastsqd(i,1)==3
        breastsqd_new(i,3)=1;
    elseif breastsqd(i,1)==4
        breastsqd_new(i,4)=1;
    else
        breastsqd_new(i,5)=1;
    end
end


%IRRADIAT

%1-NO
%2-YES

irradiat = table2array(matriz(:,10));
irradiat = grp2idx(irradiat);

%Montamos las matrices

clases_new = clases;
deg = table2array(matriz(:,7));
matriz =[age_new menopause_new tumor_new nodes_new node_new deg breast breastsqd_new irradiat];

%Procedo ha hacer la eliminacion de las matrices mas correladas


%Limpio ademas la matriz general de las clases y la primera columna (ya que
%se trata solo de un identificador)


%Calculamos la matriz de autocorrelaciones entre todas las variables para
%ver si alguna variable esta muy correlacionada con otra y asi poder
%eliminarla. Para ello voy a realizar este estudio por clases, ya que es
%mucho mas eficiente, separando en dos matrices la matriz incial segun la
%etiqueta de estas
[m1,n1] = size(clases_new);
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



%Correlaciones de ambas:

% corrM = corr(matM, matM);
% corrB = corr(matB, matB);
% 
% corrM=reshape(corrM(~diag(ones(1,size(corrM, 1)))), size(corrM)-[1 0]);
% corrB=reshape(corrB(~diag(ones(1,size(corrB, 1)))), size(corrB)-[1 0]);
% 
% detM = corrM>0.8 | corrM<-0.8;
% detB = corrB>0.8 | corrB<-0.8;
% 
% 
% 
% %eliminamos las variables que superen el umbral de correlacion
% 
% [m2,n2] = size(detM);
% 
% for i=1:m2
%     for j=1:n2
%         if detM(i,j)==1
%             matriz(:,i)=[];
%         end
%         if detB(i,j)==1
%             matriz(:,i)=[];
%         end
%     end
% end

%Procedo a extraer los rankings con los metodos empleados:
% 
% [m,n] = size(matriz);
% %fscore:
% [out, rank] = fscore(matriz,clases_new); 
% rank = rank';
% %relieff:
% [ranks,weights] = relieff(matriz,clases_new,m+1); 
% 
% %UDFS
% ranking = UDFS(matriz, clases_new);
% ranking = ranking';
% 
% %mutInfFS
% % [ rankmut , w] = mutInfFS( matriz,clases_new,10 );
% 
% %MCFS_p
% % [FeaIndex,FeaNumCandi] = MCFS_p(matriz,clases_new)
% 
% %LaPlacianScore
% [Y] = LaplacianScore(matriz, clases_new);
% [B,rankL] = sortrows(Y);
% rankL = rankL';
% 
% %infFS
% [rankInf, WEIGHTINF] = infFS( matriz, clases_new, 0.1, 0, 0 );
% 
% %ILFS
%  [rankILFS, WEIGHTILFS] = ILFS(matriz, clases_new  );
%  
% %  %fsvFS
% %  [ rankfsvFS , w] = fsvFS( matriz,clases_new,30 )
%  
% %FSASL
% % [W, S, A, objHistory] = FSASL(matriz, clases_new);
% 
% %ECFS
% % [ rankingECFS ] = ECFS( matriz, clases_new, 0.1 )
% 
% %Montamos la matriz de rankings (ordenados siendo la mayor la primera
% %columna y menos la ultima, cada fila es un metodo, el orden será el
% %siguiente):
% %1-fscore
% %2-relieff
% %3-UDFS
% %4-LaplacianScore
% %5-infFS
% %6-ILFS
% 
% mat_ranks =[rank; ranks; ranking; rankL; rankInf; rankILFS];







        
    


