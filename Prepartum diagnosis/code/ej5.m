ima=imread('T1axial.png'); ima=double(ima(:,:,1));   

%translation 
figure 
colormap(gray);  
% subplot(1,2,1),imagesc(ima);
% title('Imagen Original');
tima=imtranslate(ima,[20,20]); 
subplot(2,2,1),imagesc(tima);  
title('Imagen trasladada');
%rotacion   
% subplot(1,2,1),imagesc(ima); 
tima=imrotate(ima,-20,'bilinear','crop'); 
subplot(2,2,2),imagesc(tima);   
title('Imagen rotada');
%escalado 
% subplot(1,2,1),imagesc(ima); 
tima=imresize(ima,2*size(ima),'bilinear'); 
subplot(2,2,3),imagesc(tima); 
title('Imagen escalada');
%affine transform 
% subplot(1,2,1),imagesc(ima); 
M=[.5 0 0; .5 2 0; 0 0 1];
tform = maketform('affine',M); 
tima = imtransform(ima,tform,'bicubic'); 
subplot(2,2,4),imagesc(tima);
title('Imagen con affine');

% Combinación registros:
figure;
colormap(gray); 
subplot(1,2,1),imagesc(ima);
title('Imagen Original');
subplot(1, 2, 2);
tima=imresize(ima,2*size(ima),'bilinear');
tima=imrotate(tima,-20,'bilinear','crop'); 
tima=imtranslate(tima,[20,20]);
imagesc(tima);
title('Rotación + escalado + traslación');

%% Ejercicio 5b

fixed = dicomread('knee1.dcm'); 
moving = dicomread('knee2.dcm'); 
 
%muestra las imágenes antes de registrar 
figure 
subplot(1, 3, 1);
original = imshowpair(fixed, moving,'Scaling','joint')
title('Antes de registrar');
 
[optimizer, metric] = imregconfig('multimodal'); 
% registro 
movingRegistered1 = imregister(moving, fixed, 'affine', optimizer, metric); 
 
%muestra los resultados 
subplot(1, 3, 2); 
r1 =imshowpair(fixed, movingRegistered1,'Scaling','joint')
title('Registro lineal');
 
% si queremos la matriz de transformación podemos usar la función imregtform para estiamrla y la función imwarp para aplicarla 
tform = imregtform(moving, fixed, 'affine', optimizer, metric); 
 
movingRegistered2=imwarp(moving,tform,'OutputView',imref2d(size(fixed) )); 
 
%si queremos realizar el registro no lineal  
[D,moving_reg] = imregdemons(moving,fixed); 
subplot(1, 3, 3); 
r2 = imshowpair(fixed, moving_reg,'Scaling','joint') 
title('Registro no lineal');

correlacion1 = corrcoef(double(fixed), double(movingRegistered1));
correlacion2 = corrcoef(double(fixed), double(moving_reg));