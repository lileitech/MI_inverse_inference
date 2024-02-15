function computeTransmural(o)

if ~o.available.mesh1
    o.prepareMesh1;
end
o.printStatus('Computing transmural coordinate with RV epicardium on septum...');
t = toc;

idsEpi = o.m1.surToVol(o.m1.sur.pointData.class==2);
idsEndo = o.m1.surToVol(o.m1.sur.pointData.class==3 | o.m1.sur.pointData.class==4);
idsLVEndo = o.m1.surToVol(o.m1.sur.pointData.class==3);
idsRVsurf = o.m1.surToVol(o.m1.sur.pointData.class==4);
ucl = unique(o.m1.vol.cellData.class);
idsSept = intersect(o.m1.vol.cells(o.m1.vol.cellData.class==ucl(1),:), o.m1.vol.cells(o.m1.vol.cellData.class==ucl(2),:));
idsEpi = setdiff(idsEpi, idsSept);
idsEndo = setdiff(idsEndo, idsSept);

ids = [idsEpi; idsSept; idsEndo];
val = [zeros(size(idsEpi,1)+size(idsSept,1),1); ones(size(idsEndo))];
tmLaplace = solveLaplace(o.m1.L, ids, val, o.cfg.tol, o.cfg.maxit);

T = normalizedGradField(o.m1.G, tmLaplace, o.cfg.tol, true, o.m1.vol.points, o.m1.vol.cells);

tmDistEpi = solveTrajectDist(o.m1.G, T, [idsEpi; idsSept], zeros(size(idsEpi,1)+size(idsSept,1),1), o.cfg.tol, o.cfg.maxit);
tmDistEndo = solveTrajectDist(o.m1.G, -T, idsEndo, zeros(size(idsEndo)), o.cfg.tol, o.cfg.maxit);

o.m1.tm = tmDistEpi./(tmDistEpi+tmDistEndo);
o.m0.tm = min(max(o.m1.M*o.m1.tm,0),1);

ids = [idsEpi; idsSept];
val = [zeros(size(idsEpi)); ones(size(idsSept))];
o.m1.ridgeLaplace = solveLaplace(o.m1.L, ids, val, o.cfg.tol, o.cfg.maxit);
idsRidge = find(o.m1.ridgeLaplace > 0.7);

idsRVEpi = intersect(idsRVsurf, idsRidge);
idsRVEndo = setdiff(idsRVsurf, idsRVEpi);
ids = [idsEpi; idsRVEpi; idsRVEndo; idsLVEndo];
val = [zeros(size(idsEpi,1)+size(idsRVEpi,1),1); ones(size(idsRVEndo,1)+size(idsLVEndo,1),1)];
tmRVEpiLaplace = solveLaplace(o.m1.L, ids, val, o.cfg.tol, o.cfg.maxit);
T = normalizedGradField(o.m1.G, tmRVEpiLaplace, o.cfg.tol, true, o.m1.vol.points, o.m1.vol.cells);

tmRVEpiDistEpi = solveTrajectDist(o.m1.G, T, [idsEpi; idsRVEpi], zeros(size(idsEpi,1)+size(idsRVEpi,1),1), o.cfg.tol, o.cfg.maxit);
tmRVEpiDistEndo = solveTrajectDist(o.m1.G, -T, [idsLVEndo; idsRVEndo], zeros(size(idsLVEndo,1)+size(idsRVEndo,1),1), o.cfg.tol, o.cfg.maxit);
o.m1.tmRVEpi = tmRVEpiDistEpi./(tmRVEpiDistEpi+tmRVEpiDistEndo);
o.m0.tmRVEpi = min(max(o.m1.M*o.m1.tmRVEpi,0),1);
o.result.pointData.tm = single(o.m0.tmRVEpi);

if o.cfg.exportLevel > 1
    o.m1.debug.pointData.tmLaplace = single(tmLaplace);
    o.m1.debug.pointData.tmDistEpi = single(tmDistEpi);
    o.m1.debug.pointData.tmDistEndo = single(tmDistEndo);
    o.m1.debug.pointData.tm = single(o.m1.tm);
	o.m1.debug.pointData.tmRVEpi = single(o.m1.tmRVEpi);
    o.m1.debug.pointData.ridgeLaplace = single(o.m1.ridgeLaplace);
	disp('Writing out to debug1')
    vtkWrite(o.m1.debug, sprintf('%sdebug1.vtu', o.cfg.outPrefix));
    
    o.m0.debug.pointData.tm = single(o.m0.tm);
	o.m0.debug.pointData.tmRVEpi = single(o.m0.tmRVEpi);
    vtkWrite(o.m0.debug, sprintf('%sdebug0.vtu', o.cfg.outPrefix));
end


o.printStatus(sprintf('%.1f seconds\n', toc-t), true);
o.available.transmural = true;

end
