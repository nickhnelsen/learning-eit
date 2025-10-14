% Description of Neumann boundary condition for the
% conductivity problem.  
%
% The format of this file is suitable for describing the boundary condition
% for the assempde.m routine of Matlabs PDE toolbox.
%
% Arguments:
% p     triangulation points
% e     edge data
% u     not used here
% time  not used here
%
% Returns:
% q     zeros(1,ne), where ne is the number of edges in e
% g     values of Neumann data at centerpoint on each edge 
% h     zeros(1,2*ne)
% r     zeros(1,2*ne)
%
% Samuli Siltanen May 2008; Nicholas H. Nelsen Dec. 2021

function [q,g,h,r] = BoundaryData(p,e,~,~) 
    % Number of edges
    ne = size(e,2);

    % Give value to q, g and h
    q = zeros(1,ne);
    h = zeros(1,2*ne);
    r = zeros(1,2*ne);

    % Load current Fourier mode index/wavenumber
    load  data/BoundaryDataN n

    % Coordinates of starting and ending points of the current edge
    sp1 = p(1,e(1,:));
    sp2 = p(2,e(1,:));
    ep1 = p(1,e(2,:));
    ep2 = p(2,e(2,:));

    % Compute midpoint of boundary segment
    mp1 = (sp1+ep1)/2;
    mp2 = (sp2+ep2)/2;

    % Evaluate Neumann data at the plane point (mp1,mp2)
    % We know that this trigonometric data integrates to zero, 
    % ensuring solvability of the Neumann problem.
    g = 1/sqrt(2*pi)*exp(1i*n*angle(mp1+1i*mp2));
end

