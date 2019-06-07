#!/usr/bin/env Rscript
#args = commandArgs(trailingOnly=TRUE)
#subject_number = args[1]
#iaps_number = args[2]
#x_min = args[3]
#x_max = args[4]
#y_min = args[5]
#y_max = args[6]

#install.packages('png')
#install.packages("ggplot2")
#install.packages("devtools")
#devtools::install_github("slowkow/ggrepel")
#devtools::install_github("thomasp85/ggforce")
#devtools::install_github('thomasp85/gganimate')
#install.packages("gifski")

# fix ellipsoid problem
subject_number = "121"
iaps_number = "2190"

# For animation
library(gganimate)

ggplot(data, aes(x, y)) +
  annotation_custom(g, xmin=0, xmax=1280, ymin=0, ymax=1024) +
  xlim(0, 1280) + 
  ylim(0, 1024) +
  geom_ellipse(aes(x0 = x_center, y0 =  y_center, a = h, b = w, angle = 0),  color="orange") +
  labs(title='Samples {frame_time}') +
  geom_point(alpha=1.0, size=3.0, colour='green') +
  transition_time(z)+
  shadow_wake(wake_length=0.3, size=0.5) +
  ease_aes('linear')

plot_gaze <- function(subject_number, iaps_number) {
  # or just load from python output
  data<- read.csv(sprintf("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_%s_true_fixation.csv", subject_number,  subject_number, iaps_number))
  
  # feature enginnering for scaled X and Y gaze for both left and right eye
  colnames(data)
  
  x = data$x_interpolated
  y = data$y_interpolated
  #x = data$CursorX
  #y = data$CursorY
  #numberRows <- nrow(data)
  #new_vector <- seq(1, numberRows, by=1)
  #data$TR <- new_vector
  z = data$index
  #t = data$saccade_candidate
  i = data$final_data_type

  unique(i)
  #c = data$final_gaze_type
  
  library(png)
  img <- readPNG(sprintf("/study/midusref/DATA/Eyetracking/david_analysis/resized_IAPS/%s_resized.png", iaps_number))
  
  library(grid)
  g <- rasterGrob(img, interpolate=TRUE, height=1, width=1) 
  
  #rectangle_aoi <- read.csv(sprintf("/study4/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_%s_rectangle_aoi.csv", subject_number, subject_number, iaps_number))
  
  library(ggplot2)
  library(ggrepel)
  library(ggforce)
  library(gganimate)
  
  rectangle_aoi <- read.csv(sprintf("/study4/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_%s_rectangle_aoi.csv", subject_number, subject_number, iaps_number))
  
  xmin = rectangle_aoi$Xmin
  xmax = rectangle_aoi$Xmax
  ymin = rectangle_aoi$Ymin
  ymax = rectangle_aoi$Ymax
  
  rectangle_aoi_total <- length(xmin)

  if (rectangle_aoi_total == 1) {
    ggplot(data, aes(x=x, y=y)) +
      annotation_custom(g, xmin=0, xmax=1280, ymin=0, ymax=1024) +
      xlim(0, 1280) + 
      ylim(0, 1024) +
      geom_rect(aes(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax), color ="orange", alpha=0.01) +
      #geom_ellipsis(aes(x0 =492.8, y0 =  870.4, a = 136.533333, b = 128.0, angle = 0),  color="orange") +
      geom_point() +
      geom_path(aes(color = i == "true_fixation", group= 1)) +
      #geom_text(aes(label=ifelse(i=="saccade", z, ""))) +
      scale_color_manual(name="Gaze Type",
                         labels=c("Saccade","True_fixation", "Missing"),
                         values=alpha(c("black","red", "orange"), 1.0)) +
      labs(title=sprintf("IAPS %s Subject %s Gaze", iaps_number, subject_number)) +
      #geom_text(aes(label=ifelse(z=="0","Start Gaze", "")), size=5, color="red") +
      geom_label_repel(aes(label = ifelse(z=="0","Start Gaze", "")),
                       box.padding   = 0.55, 
                       point.padding = 0.5,
                       segment.color = 'green',
                       arrow = arrow(length = unit(0.01, "npc")))
    # give origin, a= height, b = width
    #geom_ellipsis(aes(x0 =1121.6, y0 =  498.346667, a = 97.28, b = 126.4, angle = 0),  color="green")
    #geom_ellipsis(aes(x0 = 627.2, y0 =  547.84, a = 442.026667, b = 368.0, angle = 0), color="green")
    ggsave(sprintf("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_%s_gaze_plot_rectangle.png", subject_number, subject_number, iaps_number), width = 20, height = 15, units = "cm")
  } else if (rectangle_aoi_total == 2) {
    
    xmin_one = xmin[1]
    xmax_one = xmax[1]
    ymin_one = ymin[1]
    ymax_one = ymax[1]
    
    xmin_two = xmin[2]
    xmax_two = xmax[2]
    ymin_two = ymin[2]
    ymax_two = ymax[2]
    
    ggplot(data, aes(x=x, y=y)) +
      annotation_custom(g, xmin=0, xmax=1280, ymin=0, ymax=1024) +
      xlim(0, 1280) + 
      ylim(0, 1024) +
      geom_rect(aes(xmin=xmin_one, ymin=ymin_one, xmax=xmax_one, ymax=ymax_one), color ="orange", alpha=0.01) +
      geom_rect(aes(xmin=xmin_two, ymin=ymin_two, xmax=xmax_two, ymax=ymax_two), color ="orange", alpha=0.01) +
      #geom_ellipsis(aes(x0 =492.8, y0 =  870.4, a = 136.533333, b = 128.0, angle = 0),  color="orange") +
      geom_point() +
      geom_path(aes(color = i == "true_fixation", group=1)) +
      #geom_text(aes(label=ifelse(i=="saccade", z, ""))) +
      scale_color_manual(name="Gaze Type",
                         labels=c("Saccade","True_fixation","missing"),
                         values=alpha(c("black","red", "orange"), 1.0)) +
      labs(title=sprintf("IAPS %s Subject %s Gaze", iaps_number, subject_number)) +
      #geom_text(aes(label=ifelse(z=="0","Start Gaze", "")), size=5, color="red") +
      geom_label_repel(aes(label = ifelse(z=="0","Start Gaze", "")),
                       box.padding   = 0.55, 
                       point.padding = 0.5,
                       segment.color = 'green',
                       arrow = arrow(length = unit(0.01, "npc")))
    # give origin, a= height, b = width
    #geom_ellipsis(aes(x0 =1121.6, y0 =  498.346667, a = 97.28, b = 126.4, angle = 0),  color="green")
    #geom_ellipsis(aes(x0 = 627.2, y0 =  547.84, a = 442.026667, b = 368.0, angle = 0), color="green")
    ggsave(sprintf("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_%s_gaze_plot_rectangle.png", subject_number, subject_number, iaps_number), width = 20, height = 15, units = "cm")
  } 

  ellipse_aoi <- read.csv(sprintf("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_%s_ellipse_aoi.csv", subject_number, subject_number, iaps_number))

  x_center = ellipse_aoi$Xcenter
  y_center = ellipse_aoi$Ycenter
  h = ellipse_aoi$Height
  w = ellipse_aoi$Width

  ellipse_aoi_total <- length(x_center)

  if (ellipse_aoi_total == 1) {
    ggplot(data, aes(x=x, y=y)) +
        annotation_custom(g, xmin=0, xmax=1280, ymin=0, ymax=1024) +
        xlim(0, 1280) + 
        ylim(0, 1024) +
        geom_ellipse(aes(x0 = x_center, y0 =  y_center, a = h, b = w, angle = 0),  color="orange") +
        geom_point() +
        geom_path(aes(color = i == "true_fixation", group=1)) +
        #geom_text(aes(label=ifelse(i=="saccade", z, ""))) +
        scale_color_manual(name="Gaze Type",
                           labels=c("Saccade","True_fixation","missing"),
                           values=alpha(c("black","red", "orange"), 1.0)) +
        labs(title=sprintf("IAPS %s Subject %s Gaze", iaps_number, subject_number)) +
        #geom_text(aes(label=ifelse(z=="0","Start Gaze", "")), size=5, color="red") +
        geom_label_repel(aes(label = ifelse(z=="0","Start Gaze", "")),
                         box.padding   = 0.55, 
                         point.padding = 0.5,
                         segment.color = 'green',
                         arrow = arrow(length = unit(0.01, "npc")))
      ggsave(sprintf("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_%s_gaze_plot_ellipse.png", subject_number, subject_number, iaps_number), width = 20, height = 15, units = "cm")
      
  } else if (ellipse_aoi_total == 2) {
    
    x_center = ellipse_aoi$Xcenter[1]
    y_center = ellipse_aoi$Ycenter[1]
    h = ellipse_aoi$Height[1]
    w = ellipse_aoi$Width[1]
    
    x_center_two = ellipse_aoi$Xcenter[2]
    y_center_two = ellipse_aoi$Ycenter[2]
    h_two = ellipse_aoi$Height[2]
    w_two = ellipse_aoi$Width[2]
    
    ggplot(data, aes(x=x, y=y)) +
        annotation_custom(g, xmin=0, xmax=1280, ymin=0, ymax=1024) +
        xlim(0, 1280) + 
        ylim(0, 1024) +
        geom_ellipse(aes(x0 = x_center, y0 =  y_center, a = h, b = w, angle = 0),  color="orange") +
        geom_ellipse(aes(x0 = x_center_two, y0 =  y_center_two, a = h_two, b = w_two, angle = 0),  color="orange") +
        geom_point() +
        geom_path(aes(color = i == "true_fixation", group=1)) +
        #geom_text(aes(label=ifelse(i=="saccade", z, ""))) +
        scale_color_manual(name="Gaze Type",
                           labels=c("Saccade","True_fixation","missing"),
                           values=alpha(c("black","red", "orange"), 1.0)) +
        labs(title=sprintf("IAPS %s Subject %s Gaze", iaps_number, subject_number)) +
        #geom_text(aes(label=ifelse(z=="0","Start Gaze", "")), size=5, color="red") +
        geom_label_repel(aes(label = ifelse(z=="0","Start Gaze", "")),
                         box.padding   = 0.55, 
                         point.padding = 0.5,
                         segment.color = 'green',
                         arrow = arrow(length = unit(0.01, "npc")))
      ggsave(sprintf("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_%s_gaze_plot_ellipse.png", subject_number, subject_number, iaps_number), width = 20, height = 15, units = "cm")
  }
  # # if all 4 data types (fixatoin, missing, saccade, and truefixation) need to be plotted in different color
  # # For QA purposes, the plots are displayed as true fixations vs. everything else (labeled saccaces)
  # ggplot(data, aes(x=x, y=y)) +
  #   annotation_custom(g, xmin=0, xmax=1280, ymin=0, ymax=1024) +
  #   xlim(0, 1280) + 
  #   ylim(0, 1024) +
  #   geom_rect(aes(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax), color ="orange", alpha=0.01) +
  #   #geom_ellipsis(aes(x0 =492.8, y0 =  870.4, a = 136.533333, b = 128.0, angle = 0),  color="orange") +
  #   geom_point() +
  #   geom_path(aes(color = i , group= 1)) +
  #   #geom_text(aes(label=ifelse(i=="saccade", z, ""))) +
  #   scale_color_manual(name="Gaze Type",
  #                      labels=c("Fixation","Missing", "Saccade", "True_Fixation"),
  #                      values=alpha(c("blue","orange", "black", "red"), 1.0)) +
  #   labs(title=sprintf("IAPS %s Subject %s Gaze", iaps_number, subject_number)) +
  #   #geom_text(aes(label=ifelse(z=="0","Start Gaze", "")), size=5, color="red") +
  #   geom_label_repel(aes(label = ifelse(z=="0","Start Gaze", "")),
  #                    box.padding   = 0.55, 
  #                    point.padding = 0.5,
  #                    segment.color = 'green',
  #                    arrow = arrow(length = unit(0.01, "npc")))
}
  
#plot_gaze(subject_number, iaps_number)

subject_list = c('001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '032', '033', '034', '035', '036', '038', '039', '040', '041', '042', '043', '044', '045', '047', '048', '049', '051', '052', '053', '054', '055', '056', '057', '059', '060', '061', '062', '063', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '087', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '110', '111', '113', '114', '116', '115', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '137', '136', '138', '139')
iaps_list = c('2590', '7380', '9210', '9181', '2340', '3230', '2190', '9340', '1720', '8503', '7080', '2514', '2270', '9621', '7710', '7260', '6530', '7205', '8031', '8180', '7220', '5390', '3051', '8380', '2120', '1274', '1280', '2271', '1670', '7595', '7501', '9401', '9120', '2580', '9270', '3220', '8500', '9180', '4599', '2310', '3261', '2309', '9912', '9620', '8010', '7002', '3350', '1440', '8210', '5623', '7185', '7480', '2389', '7491', '5891', '7950', '2880', '2870', '7186', '9560', '9470', '9190', '1722', '7490', '5910', '1230', '7350', '3160', '6020', '5731', '2495', '7270', '5460', '2320', '2550', '9611', '8117', '2830', '9920', '9415', '7230', '8340', '1301', '7140', '2383', '6831', '2620', '7330', '2058', '2208', '5973')

change iaps number = 2309 to 2310 and 5973 to 5972
why incorrect numbers?
#subject_list = c("001")
#iaps_list = c("1722", "2058") #2590


#plot_gaze(subject_list, iaps_list)
for (subject in subject_list) {
  for (iaps in iaps_list) {
    print (subject)
    print (iaps)
    out <- tryCatch (
    {
      plot_gaze(subject, iaps)
      print ("saving image...")
    },
    error = function(e){
      print ("not worthy of...")
    }
  )
  } 
}

### This is testing single subject
#subject_number = "001"
#iaps_number = "1722"
#data<- read.csv(sprintf("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_%s_true_fixation.csv", subject_number,  subject_number, iaps_number))

# feature enginnering for scaled X and Y gaze for both left and right eye
#colnames(data)

#x = data$x_interpolated
#y = data$y_interpolated

#ggplot(data, aes(x=x, y=y)) +
  #xlim(0, 1280) + 
  #ylim(0, 1024) +
  #geom_ellipse(aes(x0 = 892.8, y0 =  428.3733333, a = 385, b = 371.2, angle = 0),  color="orange") +
  #geom_ellipse(aes(x0 = 326.4, y0 =  496.64, a = 326, b = 350.4, angle = 0),  color="orange") 
