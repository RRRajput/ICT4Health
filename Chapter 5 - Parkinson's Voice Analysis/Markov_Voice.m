function [ ESTTR,ESTEMIT ] = Markov_Voice(status,algo)
%Trains the HMM on the set of healthy (if status = 'H')
%and unhealthy people (if status= 'P') and uses the 
% algorithm as specified by the user 
% Viterbi if algo = 'v' and Baum-Welch if algo= 'b'

vtrain =[];
for i = 0 : 4
    filename = [status '00'  int2str(i)  'a1.wav'];
    ar = PreProcess(filename);
    vtrain=[vtrain;ar];
end
if algo=='b'
    if status == 'H'
        A = mat_norm(triu(ones(8)) - triu(ones(8),2));
        A(8,8) = 0.5;
        A(8,1)=0.5;
    else
        A = mat_norm(triu(ones(8),-1) - triu(ones(8),2));
    end
    B = mat_norm(ones(8));
    [ESTTR,ESTEMIT] = hmmtrain(vtrain,A,B,'Tolerance',1e-3,'Maxiterations',200);
else
    A = mat_norm(ones(8));
    B = mat_norm(rand(8));
    [ESTTR,ESTEMIT] = hmmtrain(vtrain,A,B,'Tolerance',1e-3,'Maxiterations',200,'ALGORITHM','Viterbi');
end
end
