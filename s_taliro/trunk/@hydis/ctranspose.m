% Transpose overloaded for hydis class

% (C) 2011 by Georgios Fainekos (fainekos@asu.edu)
% Last update: 2011.06.04

function A = ctranspose(A)
A.ds = A.ds';
A.dl = A.dl';

