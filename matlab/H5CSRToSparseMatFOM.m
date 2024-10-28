function [mat] = H5CSRToSparseMatFOM(fileprefix, matrixname)

%display(['Reading ' fileprefix '...'])
ia = double(h5read(fileprefix,['/SystemMatrices' matrixname '/vecCsrIa']));
ja = double(h5read(fileprefix,['/SystemMatrices' matrixname '/vecCsrJa']));
valstruct = h5read(fileprefix,['/SystemMatrices' matrixname '/cmpCsrVal']);

val= valstruct;
val= valstruct.real.' + 1i*valstruct.imag.';

i_index =[];
tot = 0;
for i = 2:length(ia)
    lenrow = ia(i)-ia(i-1);
    if lenrow~=0
        i_index = [i_index ; (i-1)*ones(lenrow,1)];
    else
        %i_index = [i_index; (i-1)];
    end
    tot = tot +lenrow;
    
    if lenrow == 0
%         display('H')
    end
end
if length(i_index) ~= length(ja)
    error('Error in reading');
end

mat = sparse(i_index, ja, val, length(ia)-1, length(ia)-1);

end
