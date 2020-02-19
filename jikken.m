initialization

for ix=1:20
    myexperiment;
    result(ix)=numEpisode;
    restime(ix)=etime;
end
save('result(fml1).mat');