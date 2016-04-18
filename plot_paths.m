h=figure();
hold on
I=imshow('background_with_drones_video10_two_drones.png');

%M = csvread('bagfile_positionPuplisher5.csv',1);
fid = fopen('bagfile_positionPuplishe6.csv');   %// open the file
%// parse as csv, skipping the first line
contents = textscan(fid, '%f,%f,%f,%f','HeaderLines',1); 
%// unpack the fields and give them meaningful names
[~, x_pos, y_pos, ~]   = contents{:};
fclose(fid); 


hold on
plot(x_pos, y_pos,'rp')  

%M = csvread('bagfile_positionPuplisher5.csv',1);
fid = fopen('bagfile_positionPuplishe4.csv');   %// open the file
%// parse as csv, skipping the first line 
contents = textscan(fid, '%f,%f,%f,%f','HeaderLines',1); 
%// unpack the fields and give them meaningful names
[timestamp, x_pos, y_pos, quality]   = contents{:};
fclose(fid); 


%h=figure();
hold on
plot(x_pos, y_pos,'bp')
legend('Marker order 6','Marker order 4')
xlabel({'Width','(in pixels)'})
ylabel({'Height','(in pixels)'})


