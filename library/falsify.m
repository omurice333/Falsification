function [numEpisode, elapsedTime, bestRob, bestXout, bestYout, bestOpts] = falsify(config)

    function [X, TX, Y, TY, R] = yout2TY(yout)
            X = yout.getElement(3).Values.Data;
            TX = yout.getElement(3).Values.Time;
            [TX,ix,~] = unique(TX,'first');
            if size(X,1) == 1 
               X = X(ix);
            else
               X = X(:,ix);
            end
            Y = yout.getElement(2).Values.Data;
            TY = yout.getElement(2).Values.Time;
            [TY,iy,~] = unique(TY,'first');
            if size(Y,1) == 1 
               Y = Y(iy);
            else
               Y = Y(iy,:);
            end
            R = yout.getElement(1).Values.Data;
            R = R(end,1);
    end

    function [normal_preds] = normalize_pred(preds, range)
       normal_preds = [];
       lower = range(:,1);
       upper = range(:,2);
       middle = (lower + upper)/2;
       d = (upper - lower)/2;
       dd = diag(d);
       for i = 1:size(preds,2)
          normal_preds(i).str = preds(i).str;
          normal_preds(i).A = preds(i).A * dd;
          normal_preds(i).b = preds(i).b - preds(i).A * middle;
       end
    end

    function [tout, yout] = runsim(config, normal_preds)
        system_dimension = size(config.output_range, 1);
        assignin('base', 'SystemDimension', system_dimension);
        assignin('base', 'Formula', config.monitoringFormula);
        assignin('base', 'Preds', normal_preds);
        assignin('base', 'output_range', config.output_range);
      set_param([config.mdl, '/', config.agentName], 'sample_time', num2str(config.sampleTime));
      set_param([config.mdl, '/', config.agentName], 'input_range', mat2str(config.input_range));
        simOut = sim(config.mdl,'SimulationMode','normal','AbsTol','1e-5',...
                     'CaptureErrors', 'on',...
                     'SaveTime', 'on', 'TimeSaveName', 'tout',...
                     'SaveState','on','StateSaveName','xout',...
                     'SaveOutput','on','OutputSaveName','yout',...
                     'SaveFormat', 'Dataset',...
                     'StartTime', '0.0',...
                     'StopTime', num2str(config.stopTime));
        tout = simOut.get('tout');
        yout = simOut.get('yout');
    end

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

    currDir = pwd;
    addpath(currDir);
    P = py.sys.path;
    insert(P,int32(0),[pwd, '/', 'library']);
    tmpDir = tempname;
    mkdir(tmpDir);
    cd(tmpDir);
    % Load the model on the worker
    load_system(config.mdl);
    bestRob = inf;
    normal_preds = normalize_pred(config.preds, config.output_range);
    py.driver.start_learning(config.algoName,...
       size(config.output_range, 1), size(config.input_range, 1));
    tic;


    for numEpisode=1:config.maxEpisodes
        [~, yout] = runsim(config, normal_preds);
        [X, TX, Y, TY, R] = yout2TY(yout);
        rob = dp_taliro(config.targetFormula, config.preds, Y, TY, [], [], []);
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
    elapsedTime = toc;
    cd(currDir);
    rmdir(tmpDir,'s');
    rmpath(currDir);
    close_system(config.mdl, 0);
end
