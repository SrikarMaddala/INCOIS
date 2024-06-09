clc
%particle swarm optimisation
pop_size = 50;
maxt = 100;

%pso parameters 
w = 0.5;
c1 =1 ; c2 = 1;




% test cases and weights like bird pos 
% and weights

testcases = {
    [0.9701 0.1862 0.4114 0.3674 0.5781]
    [0.6123 0.5229 0.6207 0.3506 0.1315]
    [0.2190 0.7169 0.2289 0.8762 0.0675]
    [0.1204 0.4028 0.3263 0.5365 0.2407]
    [0.6376 0.1355 0.1454 0.0146 0.9627]
    };

weights = [0.3 0.2 0.1 0.15 0.25];

% number of test cases & dim
numtc = size(testcases,1);
numdim = numtc;

%pos and veocity init

positions = randi([1,numtc], numdim , pop_size);

velocity = zeros(pop_size,numdim);

%personal and global best 

pbest = positions ;
gbest = [];
gbestfitness = inf;

%pso main loop 

for itr  = 1 :maxt
    %check fitness
    fitness = calculatefitness(positions,testcases,weights);
    %update pbest 
    updatepbest = fitness < calculatefitness(positions,testcases,weights);
    pbest(updatepbest , :) = positions(updatepbest,:);

    %update gobal fitness 
    [minfitness , minindex] = min(fitness);
    if minfitness< gbestfitness
        gbest = positions(minindex , :);
        gbestfitness = minfitness;
    end

    %update position and velocity for each
    r1 = rand(pop_size,numdim);
    r2 = rand(pop_size,numdim);
    velocity= w*velocity + c1*r1 .*(pbest - positions)+c2*r2 .*(gbest - positions);
    positions = positions + velocity;
end

%dsplay best test case
disp("Best Test Case Order:");
disp(testcases(gbest,:));



%initialize fitness function
function fitness = calculatefitness(positions , testcases , weights)
    pop_size = size(positions,1);
    numtc = size(testcases,1);
    fitness = zeros (pop_size , 1);
    for p = 1:pop_size
        testcaseorder = positions(p,:);
        [~,index] = sort(testcaseorder);
        recordtc = testcases(index,:);

        %calculate  fitness 
        priority = sum((recordtc.*weights) , 2);
        fitness(p) = -sum(priority);
    end 
end    



