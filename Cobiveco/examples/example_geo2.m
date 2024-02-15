addpath('..');

%% Create cobiveco object and compute coordinates

c = cobiveco(struct('inPrefix','1000532/heart', 'outPrefix','1000532/'));
c.computeAll;
