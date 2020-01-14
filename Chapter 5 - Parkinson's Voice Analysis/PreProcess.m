function [ ar ] = PreProcess( name )
% % Preprocessing the voice sample and returning 
% % a vector of length 2000 divided in 8 states
[voice, FS] = audioread(name);
Nstates = 8;
Nsub = 5;
FSa = FS/Nsub;
Nmin = Nsub*500; %%% skipping the first 500 samples of transition phase
N = Nsub*4500;
voice = voice(Nmin:Nsub:N-Nsub);

% msq = meansqr(voice);
voice = voice/norm(voice);

t=[0:length(voice)-1]/FSa;
% plot(t,voice)

Tmin = 1./200;
[PKS,LOC] = findpeaks(voice,FSa,'MinPeakDistance',Tmin);
%%% uncomment the next two lines to print the voice samples
%%% with peaks
% figure(1)
% plot(t,voice,LOC,PKS,'or')

tt=[];
for i = 2:length(LOC)
    temp = linspace(LOC(i-1),LOC(i),Nstates+1);
    temp = temp(1:length(temp)-1);
    tt=[tt;temp'];
end


voice1=interp1(t,voice,tt);
voice1 = voice1(1:2000);

Kquant = Nstates;
amax=max(voice1);
amin=min(voice1);
delta = (amax-amin)/(Kquant-1);
ar = round((voice1-amin)/delta) + 1;
ar = ar';
end

