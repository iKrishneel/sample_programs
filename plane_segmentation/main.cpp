#include <iostream>
#include <exception>
using namespace std;

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/mouse_event.h>
#include <pcl/visualization/point_picking_event.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/vfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZRGB PointT;
vector<int> pcl_indices; //! Variable to hold the selected PCL indices
pcl::PointCloud<PointT>::Ptr gCloud (new pcl::PointCloud<PointT>);


//**********************************************************************************************************************
void saveSelectedPointCloud ();
void pointAreaCallBack (const pcl::visualization::AreaPickingEvent &, void *);
pcl::PointCloud<pcl::Normal>::Ptr computeSurfaceNormal (pcl::PointCloud<PointT>::Ptr);
void pointFeatureHistogram(const pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<pcl::Normal>::Ptr );
void viewPointFeatureHistorgram(pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<pcl::Normal>::Ptr);
void fastPointFeatureHistogram(pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<pcl::Normal>::Ptr);
Eigen::VectorXf pclModelSphere(pcl::PointCloud<PointT>::Ptr);
pcl::PointCloud<PointT>::Ptr planeSegmentation(pcl::PointCloud<PointT>::Ptr);
//**********************************************************************************************************************

int main (int argc, const char* argv[])
{
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

  try {
     //pcl::io::loadPCDFile<PointT> ("/home/krishneel/Desktop/home.pcd", *cloud);
     //pcl::io::loadPCDFile<PointT> ("/home/krishneel/Desktop/Program/readPCL/selected_cloud.pcd", *cloud);
     pcl::io::loadPCDFile<PointT> ("/home/krishneel/Desktop/Program/readPCL/tomato2.pcd", *cloud);
  } catch (exception &e){
     std::cout << "ERROR..." << e.what ()<< std::endl;
  }

  *gCloud = *cloud;
  
  Eigen::Vector3f translation = Eigen::Vector3f (0,0,0.5);
  Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity ();
  
  pcl::visualization::PCLVisualizer pclVisual ("Visual");
  pclVisual.setBackgroundColor (0,0,0);
  pclVisual.addPointCloud (cloud, "first");
  pclVisual.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "first");
  pclVisual.addCoordinateSystem (1.0);
  pclVisual.initCameraParameters ();
  

  
  //! Compute the surface normal
  pcl::PointCloud<pcl::Normal>::Ptr sNormal = computeSurfaceNormal (cloud);
  pclVisual.addPointCloudNormals<PointT, pcl::Normal> (cloud, sNormal, 10, 0.03, "normals");

  
  std::cout << "Surface Normal: " << sNormal->size()  << std::endl;
  for (int i = 0; i < sNormal->size(); i++) {
     std::cout << "Curvature: " << sNormal->points[i].curvature << "\t Axis: "
               << sNormal->points[i].normal_x << "  "
               << sNormal->points[i].normal_y << "  "
               << sNormal->points[i].normal_z << "  "
               << sNormal->points[i].curvature<<std::endl;
  } 

  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());
  normals = sNormal;
  //pointFeatureHistogram(cloud, normals);
  //fastPointFeatureHistogram(cloud, normals);
  
  //viewPointFeatureHistorgram(cloud, normals);

  
  /*Eigen::VectorXf model_coefficients =  pclModelSphere(cloud);
  pcl::ModelCoefficients sphere;
  sphere.values.resize(4);
  sphere.values[0] = model_coefficients[0];
  sphere.values[1] = model_coefficients[1];
  sphere.values[2] = model_coefficients[2];
  sphere.values[3] = model_coefficients[3];
  pclVisual.addSphere(sphere);*/
  
  
  
  while(!pclVisual.wasStopped ())
  {
     //pclVisual.showCloud (cloud);
     pclVisual.spinOnce (100);
     boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }  
  return 0;
}

//**********************************************************************************************************************
//! Function to segment the table top
//**********************************************************************************************************************\

pcl::PointCloud<PointT>::Ptr planeSegmentation(pcl::PointCloud<PointT>::Ptr cloud)
{
   pcl::PointIndicesConstPtr plane_inliers;
   pcl::ModelCoefficientsConstPtr plane_coefficients;

   /* Cloud Points belonging to the object*/
   pcl::PointCloud<PointT>::Ptr cloud_object;
   pcl::PointCloud<PointT>::Ptr plane_projected;

   pcl::ProjectInliers<PointT> proj;
   proj.setInputCloud(cloud);
   proj.setIndices(plane_inliers);
   proj.setModelCoefficients(plane_coefficients);
   proj.filter(*plane_projected);

   /* Estimation of Convex hull of projection */
   pcl::PointCloud<PointT>::Ptr plane_hull;
   pcl::ConvexHull<PointT> hull;
   hull.setInputCloud(plane_projected);
   hull.reconstruct(*plane_hull);
   

   pcl::PointIndices object_inliers;
   pcl::ExtractPolygonalPrismData<PointT> prism;
   prism.setHeightLimits(0.01, 0.5);
   prism.setInputCloud(cloud);
   prism.setInputPlanarHull(plane_hull);
   prism.segment(object_inliers);
   
   /* Extract the object point cloud */
   pcl::ExtractIndices<PointT> extract_object_indices;
   extract_object_indices.setInputCloud(cloud);
   extract_object_indices.setIndices(boost::make_shared<const pcl::PointIndices> (object_inliers));
   extract_object_indices.filter(*cloud_object);

   return (cloud_object);
}
                               
                               
//**********************************************************************************************************************
void saveSelectedPointCloud ()
{
  //cout << "Indices Size: " << pcl_indices.size () << "\t Cloud Size: " << gCloud->size ()   << endl;
  pcl::PointCloud<PointT>::Ptr nCloud (new pcl::PointCloud<PointT>);
  for (int i = 0; i < pcl_indices.size (); i++) {
    int index = pcl_indices.at (i);
    nCloud->push_back (gCloud->at (index));
    cout << "PCL Value # " << gCloud->at (index) << "\t " << nCloud->points[i]  << endl;    
  }
  cout << endl << "Indices Size: " << pcl_indices.size () << "\tCopied Size: " << nCloud->size () << endl;

  pcl::io::savePCDFileASCII ("/home/krishneel/Desktop/readPCL/selected_cloud.pcd", *nCloud);
  
}
//**********************************************************************************************************************
void pointAreaCallBack (const pcl::visualization::AreaPickingEvent &aEvent, void *viewer_void)
{
  std::vector<int> indices;
  aEvent.getPointsIndices (indices);
  pcl_indices.insert (pcl_indices.end(), indices.begin (), indices.end ());
  //std::cout << "Vector Size: " << indices.size ()   << std::endl;

  saveSelectedPointCloud ();
}

//**********************************************************************************************************************
//! Function to compute and return the surface normal of the point cloud
//**********************************************************************************************************************
pcl::PointCloud<pcl::Normal>::Ptr computeSurfaceNormal (pcl::PointCloud<PointT>::Ptr cloud)
{
  pcl::NormalEstimation<PointT, pcl::Normal>sNormal;
  sNormal.setInputCloud (cloud);

  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>());
  sNormal.setSearchMethod (tree);

  pcl::PointCloud<pcl::Normal>::Ptr cloud_normal (new pcl::PointCloud<pcl::Normal>);
  sNormal.setRadiusSearch (0.03);
  sNormal.compute (*cloud_normal);

  return cloud_normal;
}

//**********************************************************************************************************************
//! Function to compute the point feature histogram
//**********************************************************************************************************************
void pointFeatureHistogram(const pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
   pcl::PFHEstimation<PointT, pcl::Normal, pcl::PFHSignature125> pfh;
   pfh.setInputCloud(cloud);
   pfh.setInputNormals(normals);

   pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
   pfh.setSearchMethod(tree);

   pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs (new pcl::PointCloud<pcl::PFHSignature125> ());
   pfh.setRadiusSearch(0.01);
   pfh.compute(*pfhs);


   std::cout << "Size........... " << cloud->size() << "\t" << pfhs->points.size()  << std::endl;

   for (int i = 0; i < pfhs->size(); i++) {
      for(int j = 0; j < 125; j++)
      {
         std::cout << pfhs->points[i].histogram[j]  << "  " ;
      }
      std::cout  << std::endl;
   }
}


//**********************************************************************************************************************
//! Function to compute Viewpoint Feature histogram
//**********************************************************************************************************************
void viewPointFeatureHistorgram(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
   pcl::VFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> vfh;
   vfh.setInputCloud(cloud);
   vfh.setInputNormals(normals);
   pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
   vfh.setSearchMethod(tree);

   pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
   vfh.compute(*vfhs);


   for(int j = 0; j < vfhs->size(); j++)
   {
      for (int i = 0; i < 308; i++)
      {
         std::cout << vfhs->points[j].histogram[i] << "  ";
      }
   }
   std::cout << "\n\nVFH: " << vfhs->points.size() << "\t" << cloud->size()  << std::endl;   
}


//**********************************************************************************************************************
//! Function to compute fast point feature histogram
//**********************************************************************************************************************
void fastPointFeatureHistogram(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
   pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
   fpfh.setInputCloud(cloud);
   fpfh.setInputNormals(normals);

   pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
   fpfh.setSearchMethod(tree);
   pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
   fpfh.setRadiusSearch(0.01);
   fpfh.compute(*fpfhs);

   std::cout << "FPFH :" << fpfhs->points.size() << "\tCloud Size: " << cloud->size()  <<  endl;

   for (int i = 0; i < fpfhs->points.size(); i++) {
      for (int j = 0; j < 33; j++) {
         std::cout << fpfhs->points[i].histogram[j]  << "  ";
      }
      cout << endl << endl;
   }
}


//**********************************************************************************************************************
//! Function to fit a sphere model to the cloud points
//**********************************************************************************************************************
Eigen::VectorXf pclModelSphere(pcl::PointCloud<PointT>::Ptr cloud)
{
   pcl::SampleConsensusModelSphere<PointT>::Ptr modelSphere (new pcl::SampleConsensusModelSphere<PointT> (cloud, false));

   pcl::RandomSampleConsensus<PointT> ransac (modelSphere);
   Eigen::VectorXf model_coefficients;
   ransac.setDistanceThreshold(0.005);
   ransac.computeModel();
   ransac.getModelCoefficients(model_coefficients);
   
   std::cout << "Model Coefficient:\n " << model_coefficients << std::endl;
   
   return model_coefficients;
}
