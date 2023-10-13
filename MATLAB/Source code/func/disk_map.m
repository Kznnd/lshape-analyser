function V_mapped=disk_map(V1,T1,b1,V2,T2,b2,sc1,sc2)
%% Refer: https://github.com/noamaig/euclidean_orbifolds %%
% === Triangle disk orbifold ===
% Set cones around the boundary
cones1=b1(sc1);
cones2=b2(sc2);

%% =============================================
%  =======       The actual algorithm! cutting and flattening      =======
BC_1to2 = map_disks(V1,T1,cones1,V2,T2,cones2);
if find(isnan(BC_1to2))
    setGlobalx(1);
end
if getGlobalx
    V_mapped=-1;
    return;
end
V_mapped=BC_1to2*V2; 

visualization=0;
if visualization
    visual_mapping(V1,T1,V_mapped,cones1);
end
end