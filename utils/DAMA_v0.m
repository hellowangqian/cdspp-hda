function P = DAMA_v0(dataA,dataB,labelA,labelB,options)
labels = [labelA,labelB];
Ws = constructW1(labels);
optsB = options;
optsB.gnd = labelB;
Wd = 1-Ws;
Wd(labels==-1,:) = 0;
Wd(:,labels==-1) = 0;

dataA = double(dataA);
dataB = double(dataB);
[numA,dimA] = size(dataA);
[numB,dimB] = size(dataB);

W1 = constructW(dataA);
W2 = constructW(dataB);

Ds = diag(sum(Ws,2));
Dd = diag(sum(Wd,2));
D1 = diag(sum(W1,2));
D2 = diag(sum(W2,2));


Ls = Ds - Ws;
Ld = Dd - Wd;
L1 = D1 - W1;
L2 = D2 - W2;
L = [L1,zeros(numA,numB);zeros(numB,numA),L2];
Z = [dataA,zeros(numA,dimB);zeros(numB,dimA),dataB];

A = Z'*(options.mu*L+Ls)*Z;
B = Z'*Ld*Z;
B = (B+B')/2;
A = A + options.alpha*eye(size(A,2));
[P,Diag] = eigs(double(B),double(A),options.ReducedDim,'la',options);
for i = 1:size(P,2)
    if (P(1,i)<0)
        P(:,i) = P(:,i)*-1;
    end
end