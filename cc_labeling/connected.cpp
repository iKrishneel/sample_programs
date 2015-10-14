
#include "connected.h"

template<class Tin, class Tlabel,
         class Comparator, class Boolean>
int ConnectedComponents::connected(
     const Tin *img,
     Tlabel *labelimg,
     int width,
     int height,
     Comparator SAME,
     Boolean K8_connectivity) {
     label_image(img, labelimg, width, height, SAME, K8_connectivity);
     return relabel_image(labelimg, width, height);
}

template<class Tin, class Tlabel,
         class Comparator, class Boolean>
void ConnectedComponents::label_image(
    const Tin *img,
    Tlabel *labelimg,
     int width,
     int height,
     Comparator SAME,
     const Boolean K8_CONNECTIVITY) {
    const Tin *row = img;
    const Tin *last_row = 0;
    struct Label_handler {
       Label_handler(const Tin *img, Tlabel *limg) :
          piximg(img), labelimg(limg) {}
       Tlabel &operator()(const Tin *pixp) {
          return labelimg[pixp-piximg];
       }
       const Tin *piximg;
       Tlabel *labelimg;
    } label(img, labelimg);
    clear();
    label(&row[0]) = new_label();
    for (int c = 1, r = 0; c < width; ++c) {
       if (SAME(row[c], row[c-1])) {
          label(&row[c]) = label(&row[c-1]);
       } else {
          label(&row[c]) = new_label();
       }
    }
    for (int r = 1; r < height; ++r) {
       last_row = row;
       row = &img[width*r];
       if (SAME(row[0], last_row[0]))
          label(&row[0]) = label(&last_row[0]);
       else
          label(&row[0]) = new_label();

       for (int c = 1; c < width; ++c) {
          int mylab = -1;
          if (SAME(row[c], row[c-1]))
             mylab = label(&row[c-1]);
          for (int d = (K8_CONNECTIVITY ? -1 : 0); d < 1; ++d) {
             if (SAME(row[c], last_row[c+d])) {
                if (mylab >= 0) {
                    merge(mylab, label(&last_row[c+d]));
                } else {
                   mylab = label(&last_row[c+d]);
                }
             }
          }
          if (mylab >= 0) {
             label(&row[c]) = static_cast<Tlabel>(mylab);
          } else {
             label(&row[c]) = new_label();
          }
          if (K8_CONNECTIVITY && SAME(row[c-1], last_row[c])) {
             merge(label(&row[c-1]), label(&last_row[c]));
          }
       }
    }
}


template<class Tlabel>
int ConnectedComponents::relabel_image(
     Tlabel *labelimg,
     int width,
     int height) {
     int newtag = 0;
    for (int id = 0; id < labels.size(); ++id) {
       if (is_root_label(id)) {
          labels[id].tag = newtag++;
       }
    }
    for (int i = 0; i < width*height; ++i) {
       labelimg[i] = labels[root_of(labelimg[i])].tag;
    }
    return newtag;
}
