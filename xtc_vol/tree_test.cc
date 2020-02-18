/*
 * tree_test.cc
 *
 *  Created on: Feb 18, 2020
 *      Author: tonglin
 */


#include "xtc_tree.hh"
using namespace std;

int test_tree(){
    xtc_node* root = new_test_node("/");
    root->my_obj->obj_type = XTC_ROOT_GROUP;
    xtc_node* obj_0 = new_test_node("/usr");
    xtc_node* obj_1 = new_test_node("/usr/local");
    xtc_node* obj_10 = new_test_node("/usr/lib");
    xtc_node* obj_11 = new_test_node("/usr/include");
    xtc_node* obj_2 = new_test_node("/sys");
    xtc_node* obj_3 = new_test_node("/usr/local/bin");

    xtc_node* obj_4 = new_test_node("/usr/local/include");
    xtc_node* obj_5 = new_test_node("/usr/local/lib");
    xtc_node* obj_6 = new_test_node("/usr/local/include/hdf5/tools/test/h5ls");

    printf("add obj_0:\n");
    add_xtc_node(root, obj_0);
    printf("add obj_2:\n");
    add_xtc_node(root, obj_2);
    printf("add obj_1:\n");
    add_xtc_node(root, obj_1);
    printf("add obj_10:\n");
    add_xtc_node(root, obj_10);
    printf("add obj_3:\n");
    add_xtc_node(root, obj_3);
    printf("add obj_5:\n");
    add_xtc_node(root, obj_5);
    printf("add obj_6:\n");
    add_xtc_node(root, obj_6);
        printf("add obj_4:\n");
    add_xtc_node(root, obj_4);

    printf("\n\n\n");
    print_tree(root);
    return 0;
}


int main(int argc, char* argv[]){
    test_tree();
    return 0;
}
