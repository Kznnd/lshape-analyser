%Evaluate the difference between two animals
function score=evaluate_score(V1,V2,T,space_option)
if strcmp(space_option,'euclidean')
    is_p2p=true;
    % true: calculate point to point distance
    % false: calculate face center to face center distance
    if is_p2p==true
        sub=V1-V2;
        flag=zeros(size(sub,1),1);%Store vector direction
        for i=1:size(sub,1)
            %Determine positive and negative according to the direction of vector z
            if sum(sub(i,3))<0
                flag(i)=-1;
            else
                flag(i)=1;
            end
        end

        distance=sqrt(sum(sub.^2,2));
        vertor_distance=distance.*flag;
        score=mean(vertor_distance);

        % Visualize the difference between source and target with different colors
        %     color=(vertor_distance-min(vertor_distance))/...
        %         (max(vertor_distance)-min(vertor_distance))*255;
        %     figure
        %     patch('Faces',T,'Vertices',V1,'FaceColor','interp','FaceVertexCData',color,'EdgeColor','none');
        %     colorbar
        %     axis off
        %     axis equal
        %     pause(0.0001);
    else
        %calculate faces' gravity coordinate
        f_g1_x=mean([V1(T(:,1),1),V1(T(:,2),1),V1(T(:,3),1)],2);
        f_g1_y=mean([V1(T(:,1),2),V1(T(:,2),2),V1(T(:,3),2)],2);
        f_g1_z=mean([V1(T(:,1),3),V1(T(:,2),3),V1(T(:,3),3)],2);
        f_g1=[f_g1_x,f_g1_y,f_g1_z];

        f_g2_x=mean([V2(T(:,1),1),V2(T(:,2),1),V2(T(:,3),1)],2);
        f_g2_y=mean([V2(T(:,1),2),V2(T(:,2),2),V2(T(:,3),2)],2);
        f_g2_z=mean([V2(T(:,1),3),V2(T(:,2),3),V2(T(:,3),3)],2);
        f_g2=[f_g2_x,f_g2_y,f_g2_z];

        sub=f_g1-f_g2;
        flag=zeros(size(sub,1),1);%Store vector direction
        for i=1:size(sub,1)
            %Determine positive and negative according to the direction of vector z
            if sum(sub(i,3))<0
                flag(i)=-1;
            else
                flag(i)=1;
            end
        end
        distance=sqrt(sum(sub.^2,2));
        vertor_distance=distance.*flag;
        score=mean(vertor_distance);

        % Visualize the difference between source and target with different colors
        %     color=(vertor_distance-min(vertor_distance))/...
        %         (max(vertor_distance)-min(vertor_distance))*255;
        %     figure
        %     patch('Faces',T,'Vertices',V1,'FaceColor','flat','FaceVertexCData',color,'EdgeColor','none');
        %     colorbar
        %     axis off
        %     axis equal
        %     pause(0.0001);
    end
else
    %% shape basic vertor space
    score=sum(V1-V2);
end
end