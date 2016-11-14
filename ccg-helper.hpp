#ifndef CCG_HELPER_HPP_
#define CCG_HELPER_HPP_

#include <unordered_map>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "dict_map.hpp"
#include "dynet/dynet.h"
#include "dynet/model.h"

struct Sentence
{
  std::vector<std::string> raw_terms;
  std::vector<int> terms;

  std::vector<std::string> raw_lower_cased_terms;
  std::vector<int> lower_cased_terms;

  std::vector<int> poss;
  std::vector<std::vector<int>> chars;
  std::vector<int> supertags;

  // TODO: ADD REF ACTIONS
  std::vector<int> ref_actions;

  unsigned size(){
    return raw_terms.size();
  }

  std::string to_string(){
    std::string joined = boost::algorithm::join(raw_terms, " ");
    joined = joined + "\n";
    for (unsigned i = 0; i < terms.size(); ++i) {
      joined = joined + std::to_string(terms[i]) + " ";
    }
    return joined;
  }
};

void load_embedding(dynet::Dict& d, std::string pretrain_path, dynet::LookupParameter p_labels){
  std::ifstream fin(pretrain_path);  
  std::string s;
  while( getline(fin,s) )
  {   
    std::vector <std::string> fields;
    boost::algorithm::trim(s);
    boost::algorithm::split( fields, s, boost::algorithm::is_any_of( " " ) );
    std::string word = fields[0];
    std::vector<float> p_embeding;
    for (unsigned ind = 1; ind < fields.size(); ++ind){
      p_embeding.push_back(std::stod(fields[ind]));
    }
    if (d.contains(word)){
      // std::cout << "init" << std::endl;
      p_labels.initialize(d.convert(word), p_embeding);
    }
  }
}

// given the first character of a UTF8 block, find out how wide it is
// see http://en.wikipedia.org/wiki/UTF-8 for more info
inline unsigned int UTF8Len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else abort();
}

// TODO: READ NECESSARY THINGS HERE
Sentence parse_input_line(std::string line, DictMap& dict_map){
  std::istringstream in(line);

  std::string temp;
  
  Sentence sent;
  // the input looks like:
  // Mr.|NNP|N/N Vinken|NNP|N is|VBZ|(S[dcl]\NP)/NP chairman|NN|N of|IN|(NP\NP)/NP Elsevier|NNP|N/N N.V.|NNP|N ,|,|, the|DT|NP[nb]/N Dutch|NNP|N/N publishing|VBG|N/N group|NN|N .|.|.
  while(1) {
    // word:poss
    in >> temp;
    if (!in) break;
    std::string word, pos, supertag, word_pos;

    size_t p = temp.rfind('|');
    if (p == std::string::npos || p == 0 || p == (word.size()-1)) {
      std::cerr << "mal-formed POS tags: " << temp << std::endl;
      std::cerr << p << std::endl;
      abort();
    }
    word_pos = temp.substr(0,p);
    supertag = temp.substr(p+1);

    p = word_pos.rfind('|');
    word = word_pos.substr(0,p);
    pos = word_pos.substr(p+1);
    auto pds = dict_map.get_predefined_supertags();
    if (pds.find(supertag) == pds.end()){
      // replace with the dummy tag
      supertag = dict_map.get_dummy_supertag();
    }

    sent.supertags.push_back(dict_map.get_dict(DictMap::DICTMAP_IND_SUPERTAG).convert(supertag));
    sent.poss.push_back(dict_map.get_dict(DictMap::DICTMAP_IND_POS).convert(pos));
    sent.raw_terms.push_back(word);

    sent.terms.push_back(dict_map.get_dict(DictMap::DICTMAP_IND_TERM).convert(word));

    // breake the word into characters and prepare the character settings
    size_t cur = 0;
    std::vector<int> word_chars;
    while(cur < word.size()) {
      size_t len = UTF8Len(word[cur]);
      word_chars.push_back(dict_map.get_dict(DictMap::DICTMAP_IND_CHARCTER).convert(word.substr(cur,len)));
      cur += len;
    }
    sent.chars.push_back(word_chars);

    // lowercase the word
    std::string word_lc(word);
    boost::algorithm::to_lower(word_lc);
    sent.raw_lower_cased_terms.push_back(word_lc);
    sent.lower_cased_terms.push_back(dict_map.get_dict(DictMap::DICTMAP_IND_LC_TERM).convert(word_lc));
    if(!in) break;
  }
  return sent;
}

std::set<std::string> read_predefined_supertags(std::string file_path){
  std::set<std::string> tag_set;
  std::ifstream infile(file_path);
  assert(infile);
  std::string line;
  while (std::getline(infile, line)) {
    boost::algorithm::trim(line);
    if (line.length() == 0) {
      continue;
    }
    int p = line.rfind('\t');
    std::string stag = line.substr(0,p);
    tag_set.insert(stag);
    // std::cout << "read tag: " << stag << std::endl;
  }
  return tag_set;
}

std::vector<Sentence> read_from_file(std::string file_path, DictMap& dict_map)
{
  std::ifstream infile(file_path);
  assert(infile);
  std::vector<Sentence> corpus;
  std::string line;
  while (std::getline(infile, line)) {
    boost::algorithm::trim(line);
    if (line.length() == 0) {
      break;
    }
    Sentence s = parse_input_line(line, dict_map);
    corpus.push_back(s);
  }
  return corpus;
}

#endif