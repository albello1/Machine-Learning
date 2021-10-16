function [mat_ranks,typFS, num] = unsupervised_FS(matriz,n,num_carac)
%Ya tenemos la matriz prefiltrada, es hora de ordenar sus variables por el
%orden de pesos que tengan, para ello aplicaremos diferentes metodos
%basados en el aprendizaje no supervisado.

% INPUTS:
%   -matriz: Es la matriz extraída de la base de datos
%   -num_carac: Numero de variables a coger
%   -n: El método concreto de FS. Si no se especifica, se realizan todos
%       los métodos.
% OUTPUTS
%   -mat_ranks: En caso de que n sea un único numero, devolverá un vector
%   de ranks. Si es n es un vector, o se ha dejado por defecto, mat_ranks
%   será una matriz de rankings.

% En caso de dejar n por defecto(cada numero se corresponde a su valor de n):
% Montamos la matriz de rankings (ordenados siendo la mayor la primera
% columna y menos la ultima, cada fila es un metodo, el orden será el
% siguiente):
%   1-CFS
%   2-DGUFS
%   3-FSALS
%   4-LaplacianScore
%   5-LLCFS
%   6-MCFS
%   7-UDFS
%   8-UFSOL

if nargin<3
    num_carac=size(matriz,2);
    if nargin<2
        n=1:8;
    end
end

sel_caract_mcfs=[];
mat_ranks=[];

for metodo=n
   
    switch metodo
       
        case 1 %cfs:
            rank = cfs(matriz);
            ranks = rank';
            num =10;
            typFS='cfs';
     
        case 2 %dgufs:
            S = dist(matriz');
            S = -S./max(max(S)); % it's a similarity
            nClass = 3;
            alpha = 0.5;
            beta = 0.9;
            nSel = 2;
            [sel_feat_dgufs,~,~,~] = DGUFS(matriz',nClass,S,alpha,beta,nSel); 
            [~,rank]=sort(sel_feat_dgufs(:,1)+sel_feat_dgufs(:,2),'descend'); 
            ranks=rank';
            num =10;
            typFS='dgufs';
        
        case 3 %fsals
            options.LassoType = 'SLEP';
            options.SLEPrFlag = 1;
            options.SLEPreg = 0.01;
            options.LARSk = 5;
            options.LARSratio = 2;
            nClass=3;
            [W,~,~,~] = FSASL(matriz', nClass, options);
            [~,rank]=sort(abs(W(:,1))+abs(W(:,2)),'descend');
            ranks=rank';
            num =10;
            typFS='fsals';
   
        case 4 %LaPlacianScore
            W = dist(matriz');
            W = -W./max(W(:));
            [lap_scores] = LaplacianScore(matriz, W);
            [~, rankL] = sort(-lap_scores);
            ranks = rankL';
            num = subset(ranks, -lap_scores);
            typFS='LaplacianScore'; 
      
        case 5 %llcfs
            rank = llcfs(matriz);
            ranks=rank';
            num =10;
            typFS='llcfs';
       
        case 6 %mcfs
%             num_carac = 5; %Número de características a seleccionar
            options = [];
            options.k = 5;
            options.nUseEigenfunction = 4;
            [sel_caract_mcfs,~] = MCFS_p(matriz,num_carac,options);
            ranks=sel_caract_mcfs{1}';
            num =10;
            typFS='mcfs';
       
        case 7 %udfs
            nClass = 3;
            rank = UDFS(matriz, nClass);
            ranks=rank';
            num =10;
            typFS='udfs';
      
        case 8 %ufsol
            para.p0 = 'sample';
            para.p1 = 1e6;
            para.p2 = 1e2;
            nClass = 3;
            [~,~,rank,~] = UFSwithOL(matriz',nClass,para);
            ranks=rank';
            num =10;
            typFS='ufsol';
    end
    
    mat_ranks =[mat_ranks;ranks(1:num_carac)];
    
end


end