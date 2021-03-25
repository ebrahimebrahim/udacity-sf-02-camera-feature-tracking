#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <iterator>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

// actual modulus function. c++ is weird when a is negative.
inline int mod(int a, int b) {int r = a%b; return r>=0? r : r+b;}

template <typename T>
class Ringbuf {

    std::vector<T> v;
    int last_index = -1; // index of item that was most recently pushed back. -1 if none.
    size_t n = 0; // number of elements that have been pushed back (NOT memory allocated)
    size_t max_n; // size of the container, i.e. the max intended size of the ring buffer (this times sizeof(T) is the memory allocated)

  public:

    Ringbuf(size_t size) : v{}, max_n{size} {
        v.reserve(size);
    }

    size_t size() {
        return n;
    }

    bool full() {
        return n==max_n;
    }
  
    void push_back(const T & new_item) {
        if (!full()) {
            ++last_index;
            v.push_back(new_item);
            ++n;
        }
        else {
            last_index = (last_index + 1) % max_n;
            v[last_index] = new_item;
        }
    }


    struct Iterator {
        using iterator_category = std::forward_iterator_tag; // Could be a random access iterator but lets keep it simple
        using difference_type = int;
        using value_type = T;
        using pointer = T*;
        using reference = T&;

        // NOTE: ringbuf_size is the number of elements actually pushed back, so possibly less than max_n
        // i.e. it is the member variable n of Ringbuf
        Iterator(int index, std::vector<T> * vector_ptr, size_t ringbuf_size, int last_index) :
            idx{index}, v_ptr{vector_ptr},  n{ringbuf_size}, last_idx{last_index}
        {}

        reference operator*() const {return (*v_ptr)[idx];}
        pointer operator->() {return *((*v_ptr)[idx]);}
        Iterator& operator++() {
            if (idx==last_idx)
                idx=-1;
            else
                idx = (idx + 1) % n;
            return *this;
        }
        Iterator operator++(int) { Iterator tmp{*this}; ++(*this); return tmp; }
        
        friend Iterator operator-(const Iterator & lhs, difference_type j){
            Iterator output{lhs};

            // exception if lhs is end(): we treat it as the same as begin() for the perpose of subtraction
            if (output.idx==-1) 
                output.idx = (output.last_idx+1) % output.n;
            
            output.idx = mod((output.idx - j) , output.n);
            return output;
        }

        friend bool operator==(const Iterator& a, const Iterator& b) {return a.idx == b.idx;}
        friend bool operator!=(const Iterator& a, const Iterator& b) {return a.idx != b.idx;}


        private:
            int idx; // index of vector that this iterator points to. -1 will indicate the "end" state
            size_t n;
            int last_idx;
            std::vector<T> * v_ptr;

    };

    Iterator begin() { 
        if (n==0) return end();
        return Iterator((last_index+1)%n, &v, n, last_index);
    }

    Iterator end() {
        return Iterator(-1, &v, n, last_index);
    }

};

#endif /* dataStructures_h */
