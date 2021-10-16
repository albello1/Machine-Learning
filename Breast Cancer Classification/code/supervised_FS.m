function [mat_ranks, mat_nombres, num] = supervised_FS(matriz,clases_new,n)
%Ya tenemos la matriz prefiltrada, es hora de ordenar sus variables por el
%orden de pesos que tengan, para ello aplicaremos diferentes metodos
%basados en el aprendizaje supervisado, es decir, conociendo el valor de
%las etiquetas de clase.


% INPUTS:
%   -matriz: Es la matriz extraída de la base de datos
%   -clases_new: Las etiquetas de matriz
%   -n: El método concreto de FS. Si no se especifica, se realizan todos
%       los métodos.
% OUTPUTS
%   -mat_ranks: En caso de que n sea un único numero, devolverá un num
%   de ranks. Si es n es un num, o se ha dejado por defecto, mat_ranks
%   será una matriz de rankings.

% En caso de dejar n por defecto(cada numero se corresponde a su valor de n):
% Montamos la matriz de rankings (ordenados siendo la mayor la primera
% columna y menos la ultima, cada fila es un metodo, el orden será el
% siguiente):
%   1-fscore
%   2-relieff
%   3-UDFS
%   4-LaplacianScore
%   5-infFS
%   6-ILFS

if nargin<3
    n=1:6;
end

mat_ranks=[];
mat_nombres=[];

for metodo=n
    switch metodo
        case 1 %fscore:
    [w, rank] = fscore(matriz,clases_new); 
    w=w';
    ranks = rank';
    num = subset(ranks,w);
    typFS='fscore';
        case 2 %relieff:
    [m] = size(matriz,1);
    [ranks,w] = relieff(matriz,clases_new,m+1); 
    num = subset(ranks,w);
    typFS='relieff';
        case 3 %ILFS
    [ranks, w] = ILFS(matriz, clases_new  );
    num = subset(ranks,w);
    typFS='ILFS';

        case 4 %fsvFS
     [ ranks , w] = fsvFS( matriz,clases_new,30 );
     ranks = ranks';
     w=w';
     num = subset_abs(ranks,w);
     typFS='fsvFS';


    end
    
    mat_ranks =[mat_ranks;ranks];
    mat_nombres =[mat_nombres;typFS];
    
end


end
