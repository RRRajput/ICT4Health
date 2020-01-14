function [B] = mat_norm( A)
% Returns the normalized row version of the matrix A
A_sum = sum(A(:,:)');
for i = 1: length(A_sum)
   if A_sum(i) == 0
       A_sum(i) =1;
   end
end
B = A(:,:)./A_sum(:);

end

