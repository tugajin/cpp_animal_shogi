#ifndef _ALL_STATE_TABLE_H
#define _ALL_STATE_TABLE_H
#include "dobutsu.h"
#include <string>
using namespace std;

class AllStateTable {
  vUint64 contents;
  size_t c_size;
  mutable ifstream ifs;
public:
 AllStateTable(size_t size) : contents(size),c_size(size) {}
  AllStateTable(string const& fileName,bool lazy=true);
  ~AllStateTable();
  size_t size() const{ return c_size; }
  /**
   * v�����Ĥ����0�ʾ�Υ���ǥå������֤���
   * ���Ĥ���ʤ����-1���֤�
   */
  int find(uint64 v) const;
  /**
   * read only ��access�Τߤ�ͤ��롥
   */
  const uint64 operator[](size_t i) const { 
    if(contents.size()>0){
      return contents[i];
    }
    else{
      ifs.seekg(i*sizeof(uint64),ios_base::beg);
      uint64 ret;
      ifs.read((char *)&ret,sizeof(uint64));
      return ret;
    }
  }
};
#endif
