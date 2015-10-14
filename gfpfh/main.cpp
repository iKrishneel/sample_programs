#include <pcl/io/pcd_io.h>
#include <pcl/features/gfpfh.h>
 
int
main(int argc, char** argv) {
    // Cloud for storing the object.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr input(
       new pcl::PointCloud<pcl::PointXYZRGB>);
    // Object for storing the GFPFH descriptor.
    pcl::PointCloud<pcl::GFPFHSignature16>::Ptr descriptor(
       new pcl::PointCloud<pcl::GFPFHSignature16>);
 
    // Note: you should have performed preprocessing to cluster out the object
    // from the cloud, and save it to this individual file.
 
    // Read a PCD file from disk.
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(argv[1], *input) != 0) {
       return -1;
    }
    pcl::PointCloud<pcl::PointXYZL>::Ptr object(
       new pcl::PointCloud<pcl::PointXYZL>);
    for (int i = 0; i < input->size(); i++) {
       pcl::PointXYZL pt;
       pt.x = input->points[i].x;
       pt.y = input->points[i].y;
       pt.z = input->points[i].z;
       pt.label = 1;
       object->push_back(pt);
    }

    std::cout << object->size() << "\t" << input->size() << std::endl;
 
    // Note: you should now perform classification on the cloud's
    // points. See the
    // original paper for more details. For this example, we will now consider 4
    // different classes, and randomly label each point as one of them.
    // ESF estimation object.
    pcl::GFPFHEstimation<
       pcl::PointXYZL, pcl::PointXYZL, pcl::GFPFHSignature16> gfpfh;
    gfpfh.setInputCloud(object);
    // Set the object that contains the labels for each point. Thanks to the
    // PointXYZL type, we can use the same object we store the cloud in.
    gfpfh.setInputLabels(object);
    // Set the size of the octree leaves to 1cm (cubic).
    gfpfh.setOctreeLeafSize(0.01);
    // Set the number of classes the cloud has been labelled with
    // (default is 16).
    gfpfh.setNumberOfClasses(1);
    gfpfh.compute(*descriptor);

    std::cout << descriptor->size() << std::endl;
    std::cout << "Completed..." << std::endl;
}
