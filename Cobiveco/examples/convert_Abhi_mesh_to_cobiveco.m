% add the path of cobiveco
cobiveco_fold = 'E:/2022_ECG_inference/Cobiveco_IBME';
addpath(cobiveco_fold);
addpath([cobiveco_fold '/utilities']);
addpath([cobiveco_fold '/utilities/inputPreparation']);
addpath([cobiveco_fold '/functions']);
clear all; clc;

%%% ---------------input preparation----------------
foldname = 'E:\2022_ECG_inference\Cobiveco-compatible-meshes\';
foldname_new = 'E:\2022_ECG_inference\dataset_MI_inference\data_cobiveco_mesh\';

a=dir([foldname 'vol*']); 

for i=90:numel(a)  
    meshName = strrep(strrep(a(i).name, 'vol-', ''), '.mat', '');
    disp(['converting mesh ' meshName ' into cobiveco mesh...'])
    % meshName = '1010620';

    load([foldname 'vol-' meshName '.mat']);
    try
        sur = vtkDataSetSurfaceFilter(vol);
        [baseNormal,baseOrigin,~] = cobiveco_estimateBaseNormalAndOrigin(sur);
        baseShift = -7;
        baseOrigin = baseOrigin + baseShift*baseNormal;
        vol = cobiveco_clipBase(vol, baseNormal, baseOrigin);
        sur = vtkDataSetSurfaceFilter(vol);
        maxAngle = 40; % max angle of face normals wrt baseNormal for defining the base class
        numSubdiv = 2; % can help for coarse meshes (interpolation of face normals)
        [sur,debug] = cobiveco_createClasses(sur, baseNormal, maxAngle, numSubdiv);
    catch ME
        fprintf('cobiveco input preparation failed: %s\n', ME.message);
        continue;
    end


    % check whether the mesh work for converting into cobiveco
    if isfield(sur.pointData, 'class')   
        newdir=[foldname_new meshName '\'];
        if ~exist(newdir,'dir')
            mkdir(newdir);
        end
    
        vtkWrite(sur, [newdir 'heart.vtp']);
        vtkWrite(vol, [newdir 'heart.vtu']); 
        clear sur
        clear vol
        clear debug
        
        %% Create cobiveco object and compute coordinates
        try
            c = cobiveco(struct('inPrefix',[newdir 'heart'], 'outPrefix',newdir));
            c.computeAll;
          catch ME
            fprintf('cobiveco failed: %s\n', ME.message);
            continue;
        end
              
        oldname= [newdir 'heart.vtp'];
        newname= strrep(oldname, 'heart', [meshName '_heart_sur']);
        copyfile(oldname,newname);  
        delete(oldname)
        
        oldname= [newdir 'heart.vtu'];
        newname= strrep(oldname, 'heart', [meshName '_heart_tet']);
        copyfile(oldname,newname);  
        delete(oldname)
        
        oldname= [newdir 'result.vtu'];
        newname= strrep(oldname, 'result', [meshName '_heart_cobiveco']);
        copyfile(oldname,newname);  
        delete(oldname)
    
        delete([newdir 'R.mat'])

        fprintf("cobiveco worked successfully\n");
    end

end