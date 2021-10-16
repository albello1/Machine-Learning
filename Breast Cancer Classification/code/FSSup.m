function FS = FSSup(matriz, clases_new)
[m,n] = size(matriz);


%fscore:
[out, rank] = fscore(matriz,clases_new); 
rank = rank';
%relieff:
[ranks,weights] = relieff(matriz,clases_new,m+1); 


%ILFS
 [rankILFS, WEIGHTILFS] = ILFS(matriz, clases_new  );
 
%  %fsvFS
 [ rankfsvFS , w1] = fsvFS( matriz,clases_new,30 );
 rankfsvFS = rankfsvFS';
 




%Montamos la matriz de rankings (ordenados siendo la mayor la primera
%columna y menos la ultima, cada fila es un metodo, el orden será el
%siguiente):
%1-fscore
%2-relieff
%3-ILFS
%4-fsvFS


FS =[rank; ranks; rankILFS; rankfsvFS];
end

