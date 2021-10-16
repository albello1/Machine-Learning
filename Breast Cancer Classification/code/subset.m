%funcion para seleccionar el numero de features optimo
function tam = subset(ranks, w)
pesos_ord = sort(w,'descend');
total = sum(w);
deseado = total*0.95;
ca = cumsum(pesos_ord);  
lastIndex = find(ca <= deseado, 1, 'last');
B = pesos_ord(1:lastIndex);
tam = length(B);
end


