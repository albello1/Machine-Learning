imagen = imread('T1axial.png');
imagen = rgb2gray(imagen);

matricesComprobacion = [];
ac =1;
ac2=4;
figure;
colormap(gray);

for f=1:2:5
    
    h=ones(f*5,f*5)/(5*f*5*f);
    subplot(2,3,ac)
    [resultado] =Filtrar (imagen, h);
    resultado = uint8(resultado);
    ac=ac+1;
    imshow(resultado);
    title(['funcion: ',num2str(f*5)]);
    subplot(2,3,ac2)
    resu=imfilter(imagen, h);
    ac2 = ac2+1;
    imshow(resu);
    title(['imfilter: ', num2str(f*5)]);
%     matricesComprobacion(1)=resultado.-resu;
   
end

    
    
    


    