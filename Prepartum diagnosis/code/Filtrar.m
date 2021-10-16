function [resultado] =Filtrar (imagen, kernel)
[m,n]=size(kernel);
[mim,nim]=size(imagen);

radio= floor(m/2); 
resultado = zeros(mim, nim);

for i=mim:-1:1
    for j=nim:-1:1
        
       sumActual = 0;
        
       for i2=-radio:radio
           for j2=-radio:radio
               
               ni= i + i2;
               nj= j + j2;
               
               if (ni)<1
                   continue;
               end
               if(ni)>mim
                    continue;
               end
               if (nj)<1
                   continue;
               end
               if(nj)>nim
                    continue; 
               end
               comp = double(imagen(ni, nj))*kernel(i2+radio+1, j2 + radio +1);
               sumActual = sumActual + comp;
               
           end
        end
        resultado (i, j) = sumActual;    
                     
    end
end



