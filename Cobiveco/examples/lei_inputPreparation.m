addpath('../utilities');
addpath('../utilities/inputPreparation');
addpath('../functions');
addpath('../examples');
clc; clear all

%% Read original mesh

vol = vtkRead('geo3/1048965-HEART_Tet.vtk');
sur = vtkRead('geo3/1048965-HEART_sur.vtk'); 
sur_lv = vtkRead('geo3/1048965-LV.vtk');
sur_rv = vtkRead('geo3/1048965-RV.vtk');
sur_epi = vtkRead('geo3/1048965-EPI.vtk');
sur_lid = vtkRead('geo3/1048965-LID.vtk');

ids_lv = unique(int32(knnsearch(sur.points, sur_lv.points)));
ids_rv = unique(int32(knnsearch(sur.points, sur_rv.points)));
ids_epi = unique(int32(knnsearch(sur.points, sur_epi.points)));
ids_lid = unique(int32(knnsearch(sur.points, sur_lid.points)));


sur.pointData.class = ones(size(sur.points,1), 1, 'uint8');
sur.pointData.class(ids_epi) = 2;
sur.pointData.class(ids_lv) = 3;
sur.pointData.class(ids_rv) = 4;



%% Write result

vtkWrite(sur, 'geo3/heart.vtp');
vtkWrite(vol, 'geo3/heart.vtu');
