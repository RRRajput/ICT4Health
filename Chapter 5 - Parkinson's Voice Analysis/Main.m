%%%%%%% Lab08 ICT4Health
%%% The voice samples of both the healthy and unhealthy
%%% patients must be stored in the same directory
%%% as all these code files

%%% random seed 201 used because it manually
%%% gave good results for viterbi algorithm
rng(201);
%%%% Markov_Voice trains the HMM for healthy and
%%%% unhealthy people
[Trans_H,Emit_H] = Markov_Voice('H','b');
[Trans_M,Emit_M] = Markov_Voice('P','b');
acc_H=0;
count = 1;
logpsec_H = zeros(10,1);
logpsec_M = zeros(10,1);
%%%% low_limit and upper_limit decide which patients to test
%%%% the HMMs on
low_limit = 0;
upper_limit = 4;
total = upper_limit - low_limit + 1;
for i=low_limit:upper_limit
    filename = ['H00'  int2str(i)  'a1.wav'];
    ar = PreProcess(filename);
    [PStates_H,logpsec_H(count)] = hmmdecode(ar,Trans_H,Emit_H);
    [PStates_M,logpsec_M(count)] = hmmdecode(ar,Trans_M,Emit_M);
    if (logpsec_H(count) > logpsec_M(count))
        acc_H = acc_H + 1;
    end
    count = count + 1;
end
%%% accuracy on the set of healthy people
acc_H = acc_H/total

acc_M=0;
for i=low_limit:upper_limit
    filename = ['P00'  int2str(i)  'a1.wav'];
    ar = PreProcess(filename);
    [PStates_H,logpsec_H(count)] = hmmdecode(ar,Trans_H,Emit_H);
    [PStates_M,logpsec_M(count)] = hmmdecode(ar,Trans_M,Emit_M);
    if (logpsec_M(i+1) > logpsec_H(i+1))
        acc_M = acc_M + 1;
    end
    count = count+1;
end
%%% accuracy on the set of unhealthy people
acc_M = acc_M/total

%%%% overall accuracy
acc = (acc_H + acc_M)/2