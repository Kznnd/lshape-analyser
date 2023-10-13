function visual_mapping(V1,T1,V_mapped,cones1)
%% ==============================================
%  =====          Visualization parameter initialization              =====
%Some color-related variables - no need to concern yourself with these
cone_colors=[1 0.8 0;0.7 0 1; 0 0.5 0.8;0 0 0.5];
tilesize=0.00005;
maxx=-inf(length(T1),1);
maxy=-inf(length(T1),1);
for i=1:3
    maxx=max(maxx,V1(T1(:,i),1));
    maxy=max(maxy,V1(T1(:,i),2));
end
isblack=(mod(maxx,tilesize*2)<tilesize)~=(mod(maxy,tilesize*2)<tilesize);

colors=[floor(maxx/tilesize) floor(maxy/tilesize) ones(length(T1),1) ];
for i=1:3
    colors(:,i)=colors(:,i)-min(colors(:,i));
    colors(:,i)=colors(:,i)./max(colors(:,i));
end
colors(isnan(colors))=0;
colors(:,3)=1-colors(:,1);
colors=1-colors;
colors(isblack,:)=0;

%% visualization
pause(0.0001);
figure;
subplot(131);
patch('Faces',T1,'Vertices',V1,'FaceColor','flat','FaceVertexCData',colors,'edgecolor','none');
hold("on");
for i=1:length(cones1)
    scatter3(V1(cones1(i),1),V1(cones1(i),2),V1(cones1(i),3),150,cone_colors(i,:),'fill');
end
axis("equal");
axis("off");
title('Source mesh with texture');
subplot(132);
patch('Faces',T1,'Vertices',V_mapped,'FaceColor','flat','FaceVertexCData',colors,'edgecolor','none');
hold("on");
for i=1:length(cones1)
    scatter3(V_mapped(cones1(i),1),V_mapped(cones1(i),2),V_mapped(cones1(i),3),150,cone_colors(i,:),'fill');
end
axis("equal");
axis("off");
title('Target mesh textured according to map');

%Use the red line to correspond the point between two mesh
vs=V1;
vs_len_y=max(V1(:,2))-min(V1(:,2));
vt_len_y=max(V_mapped(:,2))-min(V_mapped(:,2));
len_y=max(vs_len_y,vt_len_y);
vt=V_mapped;
vt(:,2)=vt(:,2)-len_y*1.5;
subplot(133)
patch('Faces',T1,'Vertices',vs,'FaceColor','flat','FaceVertexCData',colors,'edgecolor','none');
hold("on");
patch('Faces',T1,'Vertices',vt,'FaceColor','flat','FaceVertexCData',colors,'edgecolor','none');
axis("equal");
axis("off");

show_line_percent=0.1; % Visualize the proportion of point pairs
show_point=floor(linspace(1,size(vs,1),size(vs,1)*show_line_percent));
linecolor=rand(floor(size(vs,1)*show_line_percent),3);
for j=1:floor(size(vs,1)*show_line_percent)
    line([vs(show_point(j),1),vt(show_point(j),1)],[vs(show_point(j),2),vt(show_point(j),2)],[vs(show_point(j),3),vt(show_point(j),3)],'Color',linecolor(j,:));
    scatter3(vs(show_point(j),1),vs(show_point(j),2),vs(show_point(j),3),3,'filled','MarkerFaceColor',linecolor(j,:));
    scatter3(vt(show_point(j),1),vt(show_point(j),2),vt(show_point(j),3),3,'filled','MarkerFaceColor',linecolor(j,:));
end
title('correspondence between mesh');
end