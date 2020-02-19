% (C) 2019 National Institute of Advanced Industrial Science and Technology
% (AIST)

% ARCH2014 Benchmark
%%%%%%%%%%%%%%%%%%%%



tmpl = struct();
tmpl.mdl = 'Autotrans_shift';
tmpl.input_range = [0.0 100.0; 0.0 350.0];
tmpl.output_range = [0.0 160.0;0.0 5000.0;1.0 12.0];
tmpl.init_opts = {};
tmpl.gen_opts = {};
tmpl.interpolation = {'pconst'};
tmpl.option = {};
tmpl.maxEpisodes = maxEpisodes;
tmpl.agentName = 'Falsifier';

%Formula 1
fml1 = struct(tmpl);
fml1.expName = 'fml1';
fml1.targetFormula = '[]_[0,30](g4->p1)';
fml1.monitoringDiscrate=4;
fml1.monitoringFormula = 'p1';
fml1.preds(1).str = 'p1';
fml1.preds(1).A = [1 0 0];
fml1.preds(1).b = 120.0;
fml1.preds(1).loc=[];
fml1.preds(2).str='g4';
fml1.preds(2).A=[];
fml1.preds(2).b=[];
fml1.preds(2).loc=[4];
fml1.stopTime = 30;

%Formula 1
fml2 = struct(tmpl);
fml2.expName = 'fml2';
fml2.targetFormula = '[]_[0,30](g4->p1)';
fml2.monitoringDiscrate=4;
fml2.monitoringFormula = 'p1';
fml2.preds(1).str = 'p1';
fml2.preds(1).A = [-1 0 0];
fml2.preds(1).b = -50.0;
fml2.preds(1).loc=[];
fml2.preds(2).str='g4';
fml2.preds(2).A=[];
fml2.preds(2).b=[];
fml2.preds(2).loc=[4];
fml2.stopTime = 30;



%Formula 9
fml9 = struct(tmpl);
fml9.expName = 'fml9';
fml9.targetFormula = '((g1 U g2 U g3 U g4)/\<>_[0,10](g4/\<>_[0,2](p1)))-><>_[0,10](g4->X(g4 U_[0,1]p2))';
fml9.monitoringDiscrate=4;
fml9.monitoringFormula = '<>_[0,1000](<>_[0,200]p1)-> <>_[0,1000](X(<>_[0,100]p2))';
fml9.preds(1).str = 'p1';
fml9.preds(1).A = [0 -1 0];
fml9.preds(1).b = -3500.0;
fml9.preds(1).loc=[];

fml9.preds(2).str='p2';
fml9.preds(2).A=[-1 0 0];
fml9.preds(2).b=-100;
fml9.preds(2).loc=[];

fml9.preds(3).str='g1';
fml9.preds(3).A=[];
fml9.preds(3).b=[];
fml9.preds(3).loc=[1];

fml9.preds(4).str='g4';
fml9.preds(4).A=[];
fml9.preds(4).b=[];
fml9.preds(4).loc=[4];

fml9.preds(5).str='g2';
fml9.preds(5).A=[];
fml9.preds(5).b=[];
fml9.preds(5).loc=[2];

fml9.preds(6).str='g3';
fml9.preds(6).A=[];
fml9.preds(6).b=[];
fml9.preds(6).loc=[3];
fml9.stopTime = 30;


formulas = {fml2};
configs = { };

for k = 1:size(formulas, 2)
    for i = 1:size(algorithms, 2)
        config = struct(formulas{k});
        config.algoName = algorithms{i};
        config.sampleTime = 1;
        for l = 1:maxIter
            configs = [configs, config];
        end
    end
end






config = configs{1};
config.maxEpisodes=200;
D=config.monitoringDiscrate;
 for j = 1:size(config.init_opts, 2)
       assignin('base', config.init_opts{j}{1}, config.init_opts{j}{2});
    end

    gen_opts = {};
    for k = 1:size(config.gen_opts, 2)
       range = config.gen_opts{k}{2};
       v = range(1) + (range(2) - range(1)) * rand;
       assignin('base', config.gen_opts{k}{1}, v);
       gen_opts = [gen_opts, {config.gen_opts{k}{1}, v}];
    end

for ii = 1:4
    CLG{ii} = [4+ii 8+ii];
    if ii==1
        CLG{ii+4} = ii;
    else
        CLG{ii+4} = [ii ii-1];
    end
    if ii==4
        CLG{ii+8} = ii;
    else
        CLG{ii+8} = [ii ii+1];
    end
end


currDir = pwd;
addpath(currDir);
P = py.sys.path;
insert(P,int32(0),[pwd, '/', 'library']);

load_system(config.mdl);
bestRob = inf;
normal_preds = normalize_pred(config.preds, config.output_range);
py.driver.start_learning(config.algoName,...
    size(config.output_range, 1), size(config.input_range, 1));


  alpha=100;
tic;
for numEpisode=1:config.maxEpisodes
    [~, yout] = runsim(config, normal_preds);
    [X, TX, Y, TY, R] = yout2TY(yout);
    trob = dp_taliro(config.targetFormula, config.preds, Y, TY, Y(:,3), CLG);
    dis=get(trob,1);
    con=get(trob,2);
    if isinf(con)
        rob=dis;
    else
    rob = dis + (2*exp(con/alpha)/(1+exp(con/alpha))-1);
    end
    py.driver.stop_episode_and_train(Y(end, :), exp(- R) - 1);
    disp(['Current iteration: ', num2str(numEpisode), ', rob = ', num2str(rob)])
    if rob <= bestRob
        bestRob = rob;
        bestYout = timeseries(Y, TY);
        bestXout = timeseries(X, TX);
        bestOpts = gen_opts;
        if rob < 0
            break;
        end
    end
end
etime=toc;
