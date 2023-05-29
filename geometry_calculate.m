%% Visualize generated map by SLAM algorithms
clear;
clc;
close all;
og_cloud = pcread('/home/lihaoq/Downloads/LOAM/GlobalMap.pcd');
j = 1;
figure(j)
j = j + 1;
%pcshow(og_cloud);
rotationAngles = [0 75.44 0.68922]; %75
translation = [0 0 0];
tform = rigidtform3d(rotationAngles, translation);
out_cloud = pctransform(og_cloud, tform);
figure(j)
j = j + 1;
pcshowpair(og_cloud, out_cloud);
%pcshow(out_cloud);


%% This step visualizes and calculates geomety attributes of rail tracks. 
% If machine learning is not used properly, softwares like meshlab can be
% used to discard points that are not rails.
% 
clear;
clc;
close all;
lio_sam_cloud = pcread('/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/test_data/g_test/sunset_park3/2/sunset_rail.ply');
j = 1;
figure(j)
j = j + 1;
%pcshow(lio_sam_cloud);  % Visualize original cloud

coeff = pca(lio_sam_cloud.Location);

lio_sam_cloud_pca = lio_sam_cloud.Location * coeff;

pca_cloud = pointCloud(lio_sam_cloud_pca);
figure(j)
j = j + 1;
pcshow(pca_cloud);


% Check if points are connected. If not, that means it is an outlier.
kdtree = KDTreeSearcher(pca_cloud.Location, 'Distance', 'euclidean', 'BucketSize', 50);

search_bound = size(lio_sam_cloud.Location(:,1));
tag = zeros(search_bound);
%Search each points in kdtree
Idx = rangesearch(kdtree, pca_cloud.Location, 0.15);
for i = 1 : search_bound(1)
   dimension = cell2mat(Idx(i));
   if size(dimension, 2)<5
      lio_sam_cloud_pca(i, :) = [0, 0, 0];
   end
end

pca_cloud_filtered = pointCloud(lio_sam_cloud_pca);
figure(j)
j = j + 1;
pcshow(pca_cloud_filtered)

zy_project = lio_sam_cloud_pca(:, 2:3);
% Two k means to seperate one rail from another
idx = kmeans(zy_project, 2);

rail1 = lio_sam_cloud_pca;
rail2 = lio_sam_cloud_pca;
for i = 1: search_bound(1)
   if idx(i) == 2
      rail1(i, :) = [0, 0, 0];  
   else
       rail2(i, :) = [0, 0, 0]; 
   end
end
B = [0, 0, 0];
rail1 = setdiff(rail1, B, 'rows');
rail2 = setdiff(rail2, B, 'rows');
rail1 = - rail1;
rail2 = - rail2;
rail1_pt = pointCloud(rail1);
rail2_pt = pointCloud(rail2);
figure(j)
j = j + 1;
scatter3(rail1(:,1), rail1(:, 2), rail1(:, 3), 'r', 'filled')
hold on
scatter3(rail2(:,1), rail2(:, 2), rail2(:, 3), 'b', 'filled')
legend(['rail1', 'rail2'])
hold off
xlabel('x')
ylabel('y')
zlabel('z')

%Gauge
rail1_height = max(rail1(:, 2)) - min(rail1(:, 2));
rail2_height = max(rail2(:, 2)) - min(rail2(:, 2));

%step 1. remove outlier
reg1 = polyfit(rail1(:, 1), rail1(:, 2), 1);
reg2 = polyfit(rail2(:, 1), rail2(:, 2), 1);

range1 = length(rail1);
range2 = length(rail2);

threshold = 1;

rail1_clean = remove_outlier(rail1, threshold, range1, reg1(1), reg1(2));
rail2_clean = remove_outlier(rail2, threshold, range2, reg2(1), reg2(2));

range1 = length(rail1_clean);
range2 = length(rail2_clean);

%segment into 18m each
target1 = rail1_clean(1, 1) + 18;
target2 = rail2_clean(1, 1) + 18;

segment_1 = find_segment_points(rail1_clean, range1);
segment_2 = find_segment_points(rail2_clean, range2);


for i = 2 : length(segment_1)
    rail1_segment{i - 1} = rail1_clean(segment_1{i - 1}: segment_1{i}, :);
end

for i = 2 : length(segment_2)
    rail2_segment{i - 1} = rail2_clean(segment_2{i - 1}: segment_2{i}, :);
end

top_ratio = 0.05;
bottom_ratio = 0.9;


%Linear regression on each of these segments. 

for i = 1 : length(rail1_segment)
    points = rail1_segment{i};
    points = sortrows(points, 3);
    points_top = points(1 : fix(top_ratio * length(points)), :);

    gauge_position = mean(points_top(:, 3)) - 0.015875;
    upper_bound = gauge_position + 0.02;
    lower_bound = gauge_position;

    rail_gauge = [];

    reg1_xy_quad{i} = polyfit(points(:, 1), points(:, 2), 2);
    
    kappa_arr_1 = [];
    posi_arr_1 = [];
    norm_arr_1 = [];

    for j = 1 : length(points)
        if points(j, 3) > lower_bound && points(j, 3) < upper_bound
            rail_gauge = [rail_gauge; points(j, 1), points(j, 2), points(j, 3)];
        end
        if j >= 50 && j <= length(points) - 49
            x = points([j - 49, j, j + 49], 1);
            y = polyval(reg1_xy_quad{i}, points([j-49, j, j+49], 1));
            [kappa, norm_l] = curvature(x, y);
            if kappa > 1
                kappa = 0;
            end
            posi_arr_1 = [posi_arr_1; [x(2), y(2)]];
            kappa_arr_1 = [kappa_arr_1; kappa];
            norm_arr_1 = [norm_arr_1; norm_l];
        end
    end

    kappa_arr1{i} = kappa_arr_1;
    posi_arr1{i} = posi_arr_1;
    norm_arr1{i} = norm_arr_1;

    rail1_gauge{i} = rail_gauge;

    points_bottom = points(fix(bottom_ratio * length(points)) : length(points), :);

    reg1_top_xy{i} = polyfit(points_top(:, 1), points_top(:, 2), 1);
    reg1_top_xz{i} = polyfit(points_top(:, 1), points_top(:, 3), 1);
    reg1_top_xz_quad{i} = polyfit(points_top(:, 1), points_top(:, 3), 2);

    reg1_bottom_xy{i} = polyfit(points_bottom(:, 1), points_bottom(:, 2), 1);
    reg1_bottom_xz{i} = polyfit(points_bottom(:, 1), points_bottom(:, 3), 1);

 

    %Calculate height on xz projection
    k = (reg1_top_xz{i}(1) + reg1_bottom_xz{i}(1))/2;
    rail1_height(i) = abs(reg1_top_xz{i}(2) - reg1_bottom_xz{i}(2)) / sqrt(k^2 + 1);

    for j = 1 : length(points_top)
        points_top_reg(j, :) = ...
            [points_top(j, 1), ...
            points_top(j, 1) * reg1_top_xy{i}(1) + reg1_top_xy{i}(2), ...
            points_top(j, 1) * reg1_top_xz{i}(1) + reg1_top_xz{i}(2)];

        points_top_quad_reg(j, :) = ...
            [points_top(j, 1), ...
            points_top(j, 1) * reg1_top_xy{i}(1) + reg1_top_xy{i}(2), ...
            points_top(j, 1)^2 * reg1_top_xz_quad{i}(1) + ...
            points_top(j, 1) * reg1_top_xz_quad{i}(2) + reg1_top_xz_quad{i}(3)];
        if j == fix(0.5 * length(points_top))
            rail1_profile(i) = points_top(j, 1)^2 * reg1_top_xz_quad{i}(1) + ...
                points_top(j, 1) * reg1_top_xz_quad{i}(2) + reg1_top_xz_quad{i}(3) - ...
                points_top(j, 1) * reg1_top_xz{i}(1) - reg1_top_xz{i}(2);
        end
    end
    for j = 1 : length(points_bottom)
        points_bottom_reg(j, :) = ...
            [points_bottom(j, 1), ...
            points_bottom(j, 1) * reg1_bottom_xy{i}(1) + reg1_bottom_xy{i}(2), ...
            points_bottom(j, 1) * reg1_bottom_xz{i}(1) + reg1_bottom_xz{i}(2)];
    end
    r1_seg_surface{i} = points_top_reg;  
    r1_seg_quad_surface{i} = points_top_quad_reg;
    r1_seg_bottom{i} = points_bottom_reg;
end



for i = 1 : length(rail2_segment)
    points = rail2_segment{i};
    points = sortrows(points, 3);
    points_top = points(1 : fix(top_ratio * length(points)), :);

    gauge_position = mean(points_top(:, 3)) - 0.015875;
    upper_bound = gauge_position + 0.02;
    lower_bound = gauge_position;

    rail_gauge = [];
    
    reg2_xy_quad{i} = polyfit(points(:, 1), points(:, 2), 2);
    
    kappa_arr_2 = [];
    posi_arr_2 = [];
    norm_arr_2 = [];

    for j = 1 : length(points)
        if points(j, 3) > lower_bound && points(j, 3) < upper_bound
            rail_gauge = [rail_gauge; points(j, 1), points(j, 2), points(j, 3)];
        end
        if j >= 50 && j <= length(points) - 49
            x = points([j - 49, j, j + 49], 1);
            y = polyval(reg1_xy_quad{i}, points([j-49, j, j+49], 1));
            [kappa, norm_l] = curvature(x, y);
            if kappa > 1
                kappa = 0;
            end
            posi_arr_2 = [posi_arr_2; [x(2), y(2)]];
            kappa_arr_2 = [kappa_arr_2; kappa];
            norm_arr_2 = [norm_arr_2; norm_l];
        end
    end

    kappa_arr2{i} = kappa_arr_2;
    posi_arr2{i} = posi_arr_2;
    norm_arr2{i} = norm_arr_2;

    rail2_gauge{i} = rail_gauge;

    points_bottom = points(fix(bottom_ratio * length(points)) : length(points), :);

    reg2_top_xy{i} = polyfit(points_top(:, 1), points_top(:, 2), 1);
    reg2_top_xz{i} = polyfit(points_top(:, 1), points_top(:, 3), 1);    
    reg2_top_xz_quad{i} = polyfit(points_top(:, 1), points_top(:, 3), 2);

    reg2_bottom_xy{i} = polyfit(points_bottom(:, 1), points_bottom(:, 2), 1);
    reg2_bottom_xz{i} = polyfit(points_bottom(:, 1), points_bottom(:, 3), 1);

    %Calculate height on xz projection
    k = (reg2_top_xz{i}(1) + reg2_bottom_xz{i}(1))/2;
    rail2_height(i) = abs(reg2_top_xz{i}(2) - reg2_bottom_xz{i}(2)) / sqrt(k^2 + 1);

    for j = 1 : length(points_top)
        points_top_reg(j, :) = ...
            [points_top(j, 1), ...
            points_top(j, 1) * reg2_top_xy{i}(1) + reg2_top_xy{i}(2), ...
            points_top(j, 1) * reg2_top_xz{i}(1) + reg2_top_xz{i}(2)];

        points_top_quad_reg(j, :) = ...
            [points_top(j, 1), ...
            points_top(j, 1) * reg2_top_xy{i}(1) + reg2_top_xy{i}(2), ...
            points_top(j, 1)^2 * reg2_top_xz_quad{i}(1) + ...
            points_top(j, 1) * reg2_top_xz_quad{i}(2) + reg2_top_xz_quad{i}(3)];

        if j == fix(0.5 * length(points_top))
            rail2_profile(i) = points_top(j, 1)^2 * reg2_top_xz_quad{i}(1) + ...
                points_top(j, 1) * reg2_top_xz_quad{i}(2) + reg2_top_xz_quad{i}(3) - ...
                points_top(j, 1) * reg2_top_xz{i}(1) - reg2_top_xz{i}(2);
        end
    end
    for j = 1 : length(points_bottom)
        points_bottom_reg(j, :) = ...
            [points_bottom(j, 1), ...
            points_bottom(j, 1) * reg2_bottom_xy{i}(1) + reg2_bottom_xy{i}(2), ...
            points_bottom(j, 1) * reg2_bottom_xz{i}(1) + reg2_bottom_xz{i}(2)];
    end
    r2_seg_surface{i} = points_top_reg; 
    r2_seg_quad_surface{i} = points_top_quad_reg;
    r2_seg_bottom{i} = points_bottom_reg;
end


figure
hold on

scatter3(rail1_clean(:, 1), rail1_clean(:, 2), rail1_clean(:, 3), 'r', 'filled');
scatter3(rail2_clean(:, 1), rail2_clean(:, 2), rail2_clean(:, 3), 'b', 'filled');

for i = 1:length(r1_seg_surface)
    scatter3(r1_seg_surface{i}(:, 1), r1_seg_surface{i}(:, 2), r1_seg_surface{i}(:, 3), 'b', 'filled');
    scatter3(r1_seg_bottom{i}(:, 1), r1_seg_bottom{i}(:, 2), r1_seg_bottom{i}(:, 3), 'b', 'filled');
    scatter3(r1_seg_quad_surface{i}(:, 1), r1_seg_quad_surface{i}(:, 2), r1_seg_quad_surface{i}(:, 3), 'y', 'filled');
    scatter3(rail1_gauge{i}(:, 1), rail1_gauge{i}(:, 2), rail1_gauge{i}(:, 3), 'k', 'filled');
end

for i = 1:length(r2_seg_surface)
    scatter3(r2_seg_surface{i}(:, 1), r2_seg_surface{i}(:, 2), r2_seg_surface{i}(:, 3), 'r', 'filled');
    scatter3(r2_seg_bottom{i}(:, 1), r2_seg_bottom{i}(:, 2), r2_seg_bottom{i}(:, 3), 'r', 'filled');
    scatter3(r2_seg_quad_surface{i}(:, 1), r2_seg_quad_surface{i}(:, 2), r2_seg_quad_surface{i}(:, 3), 'y', 'filled');
    scatter3(rail2_gauge{i}(:, 1), rail2_gauge{i}(:, 2), rail2_gauge{i}(:, 3), 'k', 'filled');
end



%Calculate the average y value of the entire data set. Select points that
%has the closer y value to the center as rail inner surface
y_center = mean(rail1_clean(:, 2))/2 + mean(rail2_clean(:, 2))/2;

for i = 1 : length(rail1_gauge)
    for j = 1 : length(rail1_gauge{i})
        rail1_gauge{i}(j, 4) = abs(y_center - rail1_gauge{i}(j, 2));
    end
    rail1_gauge{i} = sortrows(rail1_gauge{i}, 4);
end

for i = 1 : length(rail2_gauge)
    for j = 1 : length(rail2_gauge{i})
        rail2_gauge{i}(j, 4) = abs(y_center - rail2_gauge{i}(j, 2));
    end
    rail2_gauge{i} = sortrows(rail2_gauge{i}, 4);
end

inner_ratio = 0.3;

for i = 1 : min(length(rail1_gauge), length(rail2_gauge))
    rail1_inner = rail1_gauge{i}(1 : fix(inner_ratio * length(rail1_gauge{i})), :);
    reg1_inner_xy = polyfit(rail1_inner(:, 1), rail1_inner(:, 2), 1);
    reg1_inner_xz = polyfit(rail1_inner(:, 1), rail1_inner(:, 3), 1);
    
    rail2_inner = rail2_gauge{i}(1 : fix(inner_ratio * length(rail2_gauge{i})), :);
    reg2_inner_xy = polyfit(rail2_inner(:, 1), rail2_inner(:, 2), 1);
    reg2_inner_xz = polyfit(rail2_inner(:, 1), rail2_inner(:, 3), 1);

    k = (reg1_inner_xy(1) + reg2_inner_xy(1)) / 2;
    gauge(i) = abs(reg1_inner_xy(2) - reg2_inner_xy(2)) / sqrt(k ^ 2 + 1);
    scatter3(rail1_inner(:, 1), reg1_inner_xy(1) * rail1_inner(:, 1) + reg1_inner_xy(2), reg1_inner_xz(1) * rail1_inner(:, 1) + reg1_inner_xz(2), 'g', 'filled');
    scatter3(rail2_inner(:, 1), reg2_inner_xy(1) * rail2_inner(:, 1) + reg2_inner_xy(2), reg2_inner_xz(1) * rail2_inner(:, 1) + reg2_inner_xz(2), 'g', 'filled');
end


hold off
xlabel('x')
ylabel('y') 
zlabel('z')


%% Check result. Visualizes the label.
file_name = '/mnt/eb995bf2-ea8f-4a12-92ce-75df799099f8/rail_dataset/rail_dataset/ds0/label/labeled/1683315555.618970832_label.csv';
prediction = readmatrix([file_name]);

size = length(prediction);

bg_pred = [];
rail_pred = [];
for i = 1 : size
    if prediction(i, 4) == 0
        bg_pred = [bg_pred; prediction(i, 1), prediction(i, 2), prediction(i, 3)];
    else
        rail_pred = [rail_pred; prediction(i, 1), prediction(i, 2), prediction(i, 3)];
    end
end


figure
hold on
scatter3(bg_pred(:, 1), bg_pred(:, 2), bg_pred(:, 3), 'b', 'filled');
scatter3(rail_pred(:, 1), rail_pred(:, 2), rail_pred(:, 3), 'r', 'filled');
hold off
title(file_name)


%% Calculate curvature of each rail segment.
%Rail1 selects two points in (x, y, z) format: 点2(-67.2748, 2.94669,
%-1.35837) and 点1(-1.19652, 2.73139, -1.32479)
%Rail2: 点2(-69.1258, 1.39058, -1.35756) & 点1(-1.92143, 1.25255, -1.32207)

m1 = -1.19652 + 67.2748;
n1 = 2.73139 - 2.94669;
p1 = -1.32479 + 1.35837;

m2 = -1.92143 + 69.1258;
n2 = 1.25255 - 1.39058;
p2 = -1.32207 + 1.35756;

%Automatically calculate points about 18m apart. Then do linear regression.

line1_pt = rail1;
line2_pt = rail2;
for i = 1:search_bound(1)
    line1_pt(i, :)=[-1.19625 - m1 * i / search_bound(1), 2.73139 - n1 * i / search_bound(1), -1.32479 - p1 * i / search_bound(1)];
    line2_pt(i, :)=[-1.92143 - m2 * i / search_bound(1), 1.25255 - n2 * i / search_bound(1), -1.32207 - p2 * i / search_bound(1)];
end
%Plot each line for visualization
figure(j)
j = j + 1;
hold on
scatter3(rail1(:,1), rail1(:, 2), rail1(:, 3), 'r', '.')
scatter3(rail2(:,1), rail2(:, 2), rail2(:, 3), 'b', '.')
scatter3(line1_pt(:,1), line1_pt(:,2), line1_pt(:,3), 'y', '.')
scatter3(line2_pt(:,1), line2_pt(:,2), line2_pt(:,3), 'b', '.')
legend(['rail1', 'rail2', 'rail1_top', 'rail2_top'])
xlabel('x')
ylabel('y')
zlabel('z')
hold off



for i = 1: search_bound(1)
   
    if abs(line1_pt(i, 3) - rail1(i, 3)) > 0.05
       rail1(i, :) = [0, 0, 0];
    end
    
    if abs(line2_pt(i, 3) - rail2(i, 3)) > 0.05
       rail2(i, :) = [0, 0, 0]; 
    end
end

rail1 = sortrows(rail1);
rail2 = sortrows(rail2);
line1_pt = sortrows(line1_pt);
line2_pt = sortrows(line2_pt);

%Project onto xz to calculate distance variation
%rail1 点2(-67.2748, 2.94669, -1.35837) and 点1(-1.19652, 2.73139, -1.32479)
A1 = [-1.19652, -1.32479];
B1 = [-67.2748, -1.35837];
AB1 = B1 - A1;
ii = 1;

for i = 1 : search_bound(1)
    if rail1(i, 1) == 0
        continue;
    end
    P1 = rail1(i, [1,3]);
    %three points: A1, B1, P1
    S1(ii) = 0.5 * ((A1(1) * B1(2) - B1(1)*A1(2)) + (B1(1)*P1(2)-P1(1)*B1(2))+(P1(1)*A1(2)-A1(1)*P1(2)));
    d1(ii) = 2 * S1(ii) / norm(AB1);
    ii = ii + 1;
end


%Rail2: 点2(-69.1258, 1.39058, -1.35756) & 点1(-1.92143, 1.25255, -1.32207)
A1 = [-1.92143, -1.32207];
B1 = [-69.1258, -1.35756];
AB1 = B1 - A1;
ii = 1;
for i = 1 : search_bound(1)
    if rail2(i, 1) == 0
        continue;
    end
    P1 = rail2(i, [1,3]);
    %three points: A1, B1, P1
    S2(ii) = 0.5 * ((A1(1) * B1(2) - B1(1)*A1(2)) + (B1(1)*P1(2)-P1(1)*B1(2))+(P1(1)*A1(2)-A1(1)*P1(2)));
    d2(ii) = 2 * S2(ii) / norm(AB1);
    ii = ii + 1;
end

d1 = d1.';
d2 = d2.';

figure(j)
j = j + 1;
hold on
scatter3(rail1(:,1), rail1(:, 2), rail1(:, 3), 'r', '.')
scatter3(rail2(:,1), rail2(:, 2), rail2(:, 3), 'b', '.')
scatter3(line1_pt(:,1), line1_pt(:,2), line1_pt(:,3), 'y', '.')
scatter3(line2_pt(:,1), line2_pt(:,2), line2_pt(:,3), 'b', '.')
legend(['rail1', 'rail2', 'rail1_top', 'rail2_top'])
title('Tile removed')
xlabel('x')
ylabel('y')
zlabel('z')
hold off


%对铁轨进行分段,大约18m一段
%如果是0,去掉.如果不是0,根据值的范围划分到不同区域

rail1 = sortrows(rail1, 1);
rail2 = sortrows(rail2, 1);

rail1(all(~rail1, 2), :) = [];
rail2(all(~rail2, 2), :) = [];

rail1_size = size(rail1);
i_1 = 1; i_2 = 1; i_3 = 1;
for i = 1 : rail1_size
    if rail1(i, 1) > -18.942
        rail1_1(i_1, :) = rail1(i, :);
        i_1 = i_1 + 1;
    elseif rail1(i, 1) <=-18.942 && rail1(i, 1) > -36.0456
        rail1_2(i_2, :) = rail1(i, :);
        i_2 = i_2 + 1;
    elseif rail1(i, 1) <=-36.0456 && rail1(i, 1) > -54.8327
        rail1_3(i_3, :) = rail1(i, :);
        i_3 = i_3 + 1;
    end
end

rail2_size = size(rail2);
i_1 = 1; i_2 = 1; i_3 = 1;
for i = 1 : rail2_size
    if rail2(i, 1) > -18.4025
        rail2_1(i_1, :) = rail2(i, :);
        i_1 = i_1 + 1;
    elseif rail2(i, 1) <=-18.4025 && rail2(i, 1) > -36.378
        rail2_2(i_2, :) = rail2(i, :);
        i_2 = i_2 + 1;
    elseif rail2(i, 1) <=-36.378 && rail2(i, 1) > -54.4449
        rail2_3(i_3, :) = rail2(i, :);
        i_3 = i_3 + 1;
    end
end

figure(j)
j = j + 1;
hold on
scatter3(rail1_1(:,1), rail1_1(:,2), rail1_1(:,3), '.');
scatter3(rail1_2(:,1), rail1_2(:,2), rail1_2(:,3), '.');
scatter3(rail1_3(:,1), rail1_3(:,2), rail1_3(:,3), '.');
hold off
title('Plotting rail1 by each segment')
xlabel('x')
ylabel('y')
zlabel('z')

figure(j)
j = j + 1;
hold on
scatter3(rail2_1(:,1), rail2_1(:,2), rail2_1(:,3), '.');
scatter3(rail2_2(:,1), rail2_2(:,2), rail2_2(:,3), '.');
scatter3(rail2_3(:,1), rail2_3(:,2), rail2_3(:,3), '.');
hold off
title('Plotting rail2 by each segment')
xlabel('x')
ylabel('y')
zlabel('z')


%取出每个segment后,同样在xy投影下进行曲线拟合, 和一整个的曲线拟合进行比较
%rail1 part
figure(j)
j = j + 1;
hold on
[p1_1, S1_1] = polyfit(rail1_1(:,1), rail1_1(:,2), 2);
scatter3(rail1_1(:,1), polyval(p1_1, rail1_1(:,1)), rail1_1(:,3), '.');

[p1_2, S1_2] = polyfit(rail1_2(:,1), rail1_2(:,2), 2);
scatter3(rail1_2(:,1), polyval(p1_2, rail1_2(:,1)), rail1_2(:,3), '.');


[p1_3, S1_3] = polyfit(rail1_3(:,1), rail1_3(:,2), 2);
scatter3(rail1_3(:,1), polyval(p1_3, rail1_3(:,1)), rail1_3(:,3), '.');

[p1, S1] = polyfit(rail1(:,1), rail1(:,2), 2);

kappa_arr_1 = [];
posi_arr_1 = [];
norm_arr_1 = [];
for i = 10 : (length(rail1_1)-9)
  x = rail1_1([i-9,i,i+9], 1);
  y = polyval(p1_1, rail1_1([i-9, i, i+9], 1));
  [kappa, norm_l] = curvature(x, y);
  posi_arr_1 = [posi_arr_1;[x(2),y(2)]];
  kappa_arr_1 = [kappa_arr_1;kappa];
  norm_arr_1 = [norm_arr_1;norm_l];
end
quiver(posi_arr_1(:,1),posi_arr_1(:,2),...
    kappa_arr_1.* norm_arr_1(:,1),kappa_arr_1.* norm_arr_1(:,2))
disp('Maximum curvature in the rail1_1')
max(kappa_arr_1)


kappa_arr_2 = [];
posi_arr_2 = [];
norm_arr_2 = [];
for i = 10 : (length(rail1_2)-9)
  x = rail1_2([i-9,i,i+9], 1);
  y = polyval(p1_2, rail1_2([i-9, i, i+9], 1));
  [kappa, norm_l] = curvature(x, y);
  posi_arr_2 = [posi_arr_2;[x(2),y(2)]];
  kappa_arr_2 = [kappa_arr_2;kappa];
  norm_arr_2 = [norm_arr_2;norm_l];
end
quiver(posi_arr_2(:,1),posi_arr_2(:,2),...
    kappa_arr_2.* norm_arr_2(:,1),kappa_arr_2.* norm_arr_2(:,2))
disp('Maximum curvature in the rail1_2')
max(kappa_arr_2)


kappa_arr_3 = [];
posi_arr_3 = [];
norm_arr_3 = [];
for i = 10 : (length(rail1_3)-9)
  x = rail1_3([i-9,i,i+9], 1);
  y = polyval(p1_3, rail1_3([i-9, i, i+9], 1));
  [kappa, norm_l] = curvature(x, y);
  posi_arr_3 = [posi_arr_3;[x(2),y(2)]];
  kappa_arr_3 = [kappa_arr_3;kappa];
  norm_arr_3 = [norm_arr_3;norm_l];
end
quiver(posi_arr_3(:,1),posi_arr_3(:,2),...
    kappa_arr_3.* norm_arr_3(:,1),kappa_arr_3.* norm_arr_3(:,2))
disp('Maximum curvature in the rail1_3')
max(kappa_arr_3)


kappa_arr = [];
posi_arr = [];
norm_arr = [];
for i = 10 : (length(rail1)-9)
  x = rail1([i-9,i,i+9], 1);
  y = polyval(p1, rail1([i-9, i, i+9], 1));
  [kappa, norm_l] = curvature(x, y);
  posi_arr = [posi_arr;[x(2),y(2)]];
  kappa_arr = [kappa_arr;kappa];
  norm_arr = [norm_arr;norm_l];
end
quiver(posi_arr(:,1),posi_arr(:,2),...
    kappa_arr.* norm_arr(:,1),kappa_arr.* norm_arr(:,2))
disp('Maximum curvature in the rail1')
max(kappa_arr)

hold off


%rail2 part
figure(j)
j = j + 1;
hold on
[p2_1, S2_1] = polyfit(rail2_1(:,1), rail2_1(:,2), 2);
scatter3(rail2_1(:,1), polyval(p2_1, rail2_1(:,1)), rail2_1(:,3), '.');

[p2_2, S2_2] = polyfit(rail2_2(:,1), rail2_2(:,2), 2);
scatter3(rail2_2(:,1), polyval(p2_2, rail2_2(:,1)), rail2_2(:,3), '.');


[p2_3, S2_3] = polyfit(rail2_3(:,1), rail2_3(:,2), 2);
scatter3(rail2_3(:,1), polyval(p2_3, rail2_3(:,1)), rail2_3(:,3), '.');

[p2, S2] = polyfit(rail2(:,1), rail2(:,2), 2);

kappa_arr_1 = [];
posi_arr_1 = [];
norm_arr_1 = [];
for i = 10 : (length(rail2_1)-9)
  x = rail2_1([i-9,i,i+9], 1);
  y = polyval(p1_1, rail2_1([i-9, i, i+9], 1));
  [kappa, norm_l] = curvature(x, y);
  posi_arr_1 = [posi_arr_1;[x(2),y(2)]];
  kappa_arr_1 = [kappa_arr_1;kappa];
  norm_arr_1 = [norm_arr_1;norm_l];
end
quiver(posi_arr_1(:,1),posi_arr_1(:,2),...
    kappa_arr_1.* norm_arr_1(:,1),kappa_arr_1.* norm_arr_1(:,2))
disp('Maximum curvature in the rail2_1')
max(kappa_arr_1)


kappa_arr_2 = [];
posi_arr_2 = [];
norm_arr_2 = [];
for i = 10 : (length(rail2_2)-9)
  x = rail2_2([i-9,i,i+9], 1);
  y = polyval(p2_2, rail2_2([i-9, i, i+9], 1));
  [kappa, norm_l] = curvature(x, y);
  posi_arr_2 = [posi_arr_2;[x(2),y(2)]];
  kappa_arr_2 = [kappa_arr_2;kappa];
  norm_arr_2 = [norm_arr_2;norm_l];
end
quiver(posi_arr_2(:,1),posi_arr_2(:,2),...
    kappa_arr_2.* norm_arr_2(:,1),kappa_arr_2.* norm_arr_2(:,2))
disp('Maximum curvature in the rail2_2')
max(kappa_arr_2)


kappa_arr_3 = [];
posi_arr_3 = [];
norm_arr_3 = [];
for i = 10 : (length(rail2_3)-9)
  x = rail2_3([i-9,i,i+9], 1);
  y = polyval(p2_3, rail2_3([i-9, i, i+9], 1));
  [kappa, norm_l] = curvature(x, y);
  posi_arr_3 = [posi_arr_3;[x(2),y(2)]];
  kappa_arr_3 = [kappa_arr_3;kappa];
  norm_arr_3 = [norm_arr_3;norm_l];
end
quiver(posi_arr_3(:,1),posi_arr_3(:,2),...
    kappa_arr_3.* norm_arr_3(:,1),kappa_arr_3.* norm_arr_3(:,2))
disp('Maximum curvature in the rail2_3')
max(kappa_arr_3)


kappa_arr = [];
posi_arr = [];
norm_arr = [];
for i = 10 : (length(rail2)-9)
  x = rail2([i-9,i,i+9], 1);
  y = polyval(p2, rail2([i-9, i, i+9], 1));
  [kappa, norm_l] = curvature(x, y);
  posi_arr = [posi_arr;[x(2),y(2)]];
  kappa_arr = [kappa_arr;kappa];
  norm_arr = [norm_arr;norm_l];
end
quiver(posi_arr(:,1),posi_arr(:,2),...
    kappa_arr.* norm_arr(:,1),kappa_arr.* norm_arr(:,2))
disp('Maximum curvature in the rail2')
max(kappa_arr)

hold off

figure(j)
j = j + 1;
histogram(d1)
title('rail1 deviation to the approximated rail surface in z axis / height')


figure(j)
j = j + 1;

histogram(d2)
title('rail2 deviation to the approximated rail surface in z axis / height')



%叉乘求高, 3d下的Euclidean distance
%xy投影下的横向距离
%rail1
%rail1 点2(-67.2748, 2.94669, -1.35837) and 点1(-1.19652, 2.73139, -1.32479)
rail1_length = size(rail1, 1);
A1 = [-1.19652, 2.73139, -1.32479];
B1 = [-67.2748, 2.94669, -1.35837];
A1_xy = [-1.19652, 2.73139];
B1_xy = [-67.2748, 2.94669];
AB1_xy = B1_xy - A1_xy;pointCloud
norm_AB1_xy = norm(AB1_xy);
AB1 = B1 - A1;
norm_AB1 = norm(AB1);
for i = 1 : rail1_length
   if rail1(i, 1) == 0 
       continue;
   end
   P1 = rail1(i, :);
   AP1 = P1 - A1;
   S1 = norm(cross(AB1, AP1));
   d1_3(i) = S1 / norm_AB1;
   P1_xy = rail1(i, 1:2);
   S1_xy = 0.5 * ((A1_xy(1) * B1_xy(2) - B1_xy(1)*A1_xy(2)) + (B1_xy(1)*P1_xy(2)-P1_xy(1)*B1_xy(2))+(P1_xy(1)*A1_xy(2)-A1_xy(1)*P1_xy(2)));
   d1_xy(i) = 2 * S1_xy / norm_AB1_xy;
end
d1_3 = d1_3.';
d1_xy = d1_xy.';

%rail2
%Rail2: 点2(-69.1258, 1.39058, -1.35756) & 点1(-1.92143, 1.25255, -1.32207)
rail2_length = size(rail2, 1);
A2 = [-1.92143, 1.25255, -1.32207];
B2 = [-69.1258, 1.39058, -1.35756];
A2_xy = [-1.92143, 1.25255];
B2_xy = [-69.1258, 1.39058];
AB2_xy = B2_xy-A2_xy;
norm_AB2_xy = norm(AB2_xy);
AB2 = B2 - A2;
norm_AB2 = norm(AB2);
for i = 1 : rail2_length
   if rail2(i, 1) == 0 
       continue;
   end
   P2 = rail2(i, :);
   AP2 = P2 - A2;
   S2 = norm(cross(AB2, AP2));
   d2_3(i) = S2 / norm_AB2;
   P2_xy = rail2(i, 1:2);
   S2_xy = 0.5 * ((A2_xy(1) * B2_xy(2) - B2_xy(1)*A2_xy(2)) + (B2_xy(1)*P2_xy(2)-P2_xy(1)*B2_xy(2))+(P2_xy(1)*A2_xy(2)-A2_xy(1)*P2_xy(2)));
   d2_xy(i) = 2 * S2_xy / norm_AB1_xy;
end
d2_3 = d2_3.';
d2_xy = d2_xy.';

figure(j)
j = j + 1;
histogram(d1_3)
title('Distrubtion of Rail1 point distance to the end-to-end line in 3D')
figure(j)
j = j + 1;
histogram(d2_3)
title('Distrubtion of Rail2 point distance to the end-to-end line in 3D')
figure(j)
j = j + 1;
histogram(d1_xy)
title('Distrubtion of Rail1 point distance to the end-to-end line in xy-projection')
figure(j)
j = j + 1;
histogram(d2_xy)
title('Distrubtion of Rail2 point distance to the end-to-end line in xy-projection')

%The xy-projection distance should be around 0.65m



%Function definitions
function [kappa,norm_k] = curvature(x,y)

    x = reshape(x,3,1);
    y = reshape(y,3,1);
    t_a = norm([x(2)-x(1),y(2)-y(1)]);
    t_b = norm([x(3)-x(2),y(3)-y(2)]);
    
    M =[[1, -t_a, t_a^2];
        [1, 0,    0    ];
        [1,  t_b, t_b^2]];

    a = M\x;
    b = M\y;

    kappa  = abs(2.*(a(3)*b(2)-b(3)*a(2)) / (a(2)^2.+b(2)^2.)^(1.5));
    norm_k =  [b(2),-a(2)]/sqrt(a(2)^2.+b(2)^2.);
end


function [cloud_out] = remove_outlier(cloud_in, threshold, range, coef1, intercept)
    for i = 1:range
        d = shortest_distance(cloud_in(i, 1), cloud_in(i, 2), coef1, -1, intercept);
        if d > threshold
            cloud_in(i, :) = [0, 0, 0];
        end
    end
    B = [0, 0, 0];
    cloud_out = setdiff(cloud_in, B, "rows");
end

function [d] = shortest_distance(x1, y1, a, b, c)
    d = abs((a * x1 + b * y1 + c)/sqrt(a * a + b * b));
end

function [segment_point_list] = find_segment_points(cloud_in, range)
    i = 1;
    append_array = {};
    append_array = [append_array, i];
    while i < range
        target = cloud_in(i, 1) + 18;
        [minValue, idx] = min(abs(target - cloud_in(:, 1)));
        disp(idx);
        disp(cloud_in(idx, :));
        append_array = [append_array, idx];
        i = idx;
    end
    segment_point_list = append_array;
end