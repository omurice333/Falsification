% (C) 2019 National Institute of Advanced Industrial Science and Technology
% (AIST)

% Configurations
%%%%%%%%%%%%%%%
py.sys.setdlopenflags(int32(10));
py.importlib.import_module('ssl');
staliro_dir = 's_taliro/trunk';
maxEpisodes = 200;
maxIter = 1;
algorithms = {'A3C'};
addpath('library', 'benchmarks/transmission', 'benchmarks/powertrain',...
    'benchmarks/SteamCondenser', 'benchmarks/chasing-cars',...
    'benchmarks/neural', 'benchmarks/wind-turbine');

% Initialization
%%%%%%%%%%%%%%%%
if exist('dp_taliro.m', 'file') == 0
    addpath(staliro_dir);
    cwd = pwd;
    cd(staliro_dir);
    setup_staliro;
    cd(cwd);
end

if exist('setup_monitor.m', 'file') == 0
    addpath(fullfile(staliro_dir, 'monitor'));
    cwd = pwd;
    cd(fullfile(staliro_dir, 'monitor'));
    setup_monitor;
    cd(cwd);
end
