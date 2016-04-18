%M = csvread('bagfile_positionPuplisher5.csv',1);
fid = fopen('bagfile_positionPuplisher6.csv');   %// open the file
%// parse as csv, skipping the first line
contents = textscan(fid, '%f,%f,%f,%f','HeaderLines',1); 
%// unpack the fields and give them meaningful names
[timestamp, x_pos, y_pos, quality]   = contents{:};
fclose(fid); 


h=figure();
%I=imshow('background_with_drones.png');
hold on
plot(x_pos, y_pos,'rp')  

