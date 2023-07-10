%%求平均形状
function V_avg=generate_avg_shape(V,fnum,Vnum)
ALL_v = zeros(Vnum,3,fnum);
for i=1:fnum
    vt=V{i};
    ALL_v(:,:,i) = vt;
end
V_avg = nanmean(ALL_v,3);
end