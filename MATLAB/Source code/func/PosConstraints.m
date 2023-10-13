classdef PosConstraints<handle
    %object to aggregate positional constraints and generate the
    %corresponding matrix and RHS
    properties
        A; %the constraint matrix
        b=[]; %the RHS
        
    end
    
    methods
        function obj=PosConstraints(nvars)
            %nvars - how many vertices are there in the mesh (to set width
            %of the resulting matrix)
            obj.A=sparse(0,nvars*2);
        end
        function addConstraint(obj,ind,w,rhs)
            % adds a positional constraint on a vertex x_ind, 
            % so that x_ind*w=rhs 
            assert(length(rhs)==2);
            obj.A(end+1,ind*2-1)=w;
            obj.A(end+1,ind*2)=w;
            
            obj.b=[obj.b;rhs];
        end

        function addLineConstraint(obj,ind,n,offset)
            %add constraint for x_ind to lie on an infinite line, according
            %to a normal and offset, so that <x_ind,n>=offset
            obj.A(end+1,ind*2-1:ind*2)=n;
            
            obj.b=[obj.b;offset];
        end
    end
end